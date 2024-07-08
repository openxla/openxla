// TODO: add copyright 

#include "xla/service/experimental/auto_parallel.h"

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/errors.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/service/sharding_propagation.h"

#include <stdint.h>

// assuming two dimensional mesh heirarchy of nodes and GPUs within nodes
#define NUM_MESH_DIM 2  /* number of dimensions in the mesh grid */
#define MESH_X_DIM 2 /* number of nodes */
#define MESH_Y_DIM 4 /* number of gpus per node */
#define DEVICE_COUNT (MESH_X_DIM * MESH_Y_DIM) /* total number of devices */

namespace xla {

namespace {

  /*********************************************************/
  /* Debugging                                             */
  /*********************************************************/

  std::string LOG_HEADER(int x, const char c[]="AutoParallel: ") {
    return ((x == 0) ? (c) : ((LOG_HEADER(x - 1, c)) + "\t"));
  }

  void PrintProtoList(std::function<int()> length_fn, std::function<int64_t(int)> getter, int depth=3, std::string list_name="array") {
    int n = length_fn();

    std::string s = "";

    for (int i = 0; i < n; i++) {
      s += std::to_string(getter(i)) + " ";
    }

    VLOG(5) << LOG_HEADER(depth, "InstInfo: ") << list_name << s;

    return; 
  }

  void PrintShardingInfo(OpSharding sharding, int depth=3) {

    VLOG(5) << LOG_HEADER(depth + 1, "SharInfo: ") << "sharding: ";

    std::function<int64_t(int)> getter = [&sharding](int index) {
      return (sharding.*static_cast<int64_t (OpSharding::*)(int) const>(&OpSharding::tile_assignment_dimensions))(index);
    };

    PrintProtoList(
      std::bind(&OpSharding::tile_assignment_dimensions_size, &sharding),
      getter,
      depth + 1, "tile_assignment_dimensions:"
    );

    getter = [&sharding](int index) {
      return (sharding.*static_cast<int64_t (OpSharding::*)(int) const>(&OpSharding::tile_assignment_devices))(index);
    };

    PrintProtoList(
      std::bind(&OpSharding::tile_assignment_devices_size, &sharding),
      getter,
      depth + 1, "tile_assignment_devices:"
    );

    getter = [&sharding](int index) {
      return (sharding.*static_cast<int64_t (OpSharding::*)(int) const>(&OpSharding::iota_reshape_dims))(index);
    };

    PrintProtoList(
      std::bind(&OpSharding::iota_reshape_dims_size, &sharding),
      getter,
      depth + 1, "iota_reshape_dims:"
    );

    getter = [&sharding](int index) {
      return (sharding.*static_cast<int32_t (OpSharding::*)(int) const>(&OpSharding::iota_transpose_perm))(index);
    };

    PrintProtoList(
      std::bind(&OpSharding::iota_transpose_perm_size, &sharding),
      getter,
      depth + 1, "iota_transpose_perm:"
    );
  }

  void PrintInstructionInfo(HloInstruction* instruction, int depth=3) {

    int64_t num_operands = instruction->operand_count();
    VLOG(5) << LOG_HEADER(depth, "InstInfo: ") << "Name: " << instruction->name() << " " << instruction;
    VLOG(5) << LOG_HEADER(depth + 1, "InstInfo: ") << "num operands: " << num_operands;
    VLOG(5) << LOG_HEADER(depth + 1, "InstInfo: ") << "sharded: " << instruction->has_sharding();
    VLOG(5) << LOG_HEADER(depth + 1, "InstInfo: ") << "shape: " << instruction->shape().ToString();

    if (instruction->has_sharding()) {
      // convert to Proto and print out proto elements
      PrintShardingInfo(instruction->sharding_ptr()->ToProto(), depth + 1);
    }

    HloInstruction::InstructionVector operands = instruction->operands();
    for (int i = 0; i < num_operands; i++) {
      VLOG(5) << LOG_HEADER(depth, "InstInfo: ") << i << ": " << operands[i]->name() << " " << operands[i]->shape().ToString() << " " << operands[i];
    }

    return;
  }

  void PrintComputationInfo(HloComputation* computation, int depth=3) {
    VLOG(5) << LOG_HEADER(depth, "CompInfo: ") << "Name: " << computation->name() << " " << computation;
    VLOG(5) << LOG_HEADER(depth, "CompInfo: ") << "Instruction Count: " << computation->instruction_count();

    for (HloInstruction* instr : computation->instructions()) {
      PrintInstructionInfo(instr, depth + 1);
    }
  }

  void PrintModuleInfo(HloModule* module, int depth=1) {

    VLOG(5) << LOG_HEADER(depth, "ModuInfo: ") << "Name: " << module->name() << " " << module;
    VLOG(5) << LOG_HEADER(depth, "ModuInfo: ") << "Computation count: " << module->computation_count();

    for (HloComputation* computation : module->computations()) {
      PrintComputationInfo(computation, depth + 1);
    }

  }

  /*********************************************************/
  /* Convert instructions to modules                       */
  /*********************************************************/

  // clones a parameter instruction specifically 
  // for single-instruction HloComputations
  std::unique_ptr<HloInstruction> CloneParameterInstruction(
      HloParameterInstruction* instruction) {

    // create parameter-retrieving instruction 
    // with same shape, cloned name, param_no of 0
    Shape s = instruction->shape();
    absl::string_view name = instruction->name();

    return std::move(HloInstruction::CreateParameter(0, s, name));
  }

  // fixes instructions so that it can be the only one inside of a computation
  std::unique_ptr<HloInstruction> CloneSingleInstruction(
      HloInstruction* instruction) {

    std::unique_ptr<HloInstruction> result;

    // choose appropriate correction based on instruction type
    switch (instruction->opcode()) {
      case HloOpcode::kParameter: {
        result = CloneParameterInstruction(
          Cast<HloParameterInstruction>(instruction));
        break;
      }
      default: {
        result = instruction->Clone();
        break;
      }
    }

    return result; 
  }
  
  // Creates a module from a single instruction for running a simple pass on
  std::unique_ptr<HloModule> CreateModuleFromInstruction(
      HloInstruction* instruction) {

    // copy the instruction so as not to modify the HloModule
    std::unique_ptr<HloInstruction> instr_clone 
      = std::move(CloneSingleInstruction(instruction));
    
    // create entry computation from the single instruction
    HloComputation::Builder builder{"single-instr"};
    HloInstruction* instrp = builder.AddInstruction(std::move(instr_clone));
    std::unique_ptr<HloComputation> computation = builder.Build(instrp);

    // construct the module's configuration
    HloModuleConfig config{computation->ComputeProgramShape()};

    // construct the module from the computation 
    // (unique ptr so cleared out of memory)
    std::unique_ptr<HloModule> module = 
      std::make_unique<HloModule>(std::string(instruction->name()), config);
    module->AddEntryComputation(std::move(computation));

    // create a copy so it is completely separate from original module
    std::unique_ptr<HloModule> module_clone = module->Clone(); 

    return module_clone;
  }

  /*********************************************************/
  /* InstructionSharding Class                             */
  /*********************************************************/

  class InstructionSharding {
  public:
    InstructionSharding() = default;
    ~InstructionSharding() = default;
    InstructionSharding(const InstructionSharding &s) = default;
    
    // cost getters and setters
    uint64_t cost() const { return cost_; }
    void set_cost(uint64_t cost) { cost_ = cost; }

    // modifying the operand_shardings
    // TODO: accept a shared pointer
    void AddOpSharding(HloSharding sharding);
    std::shared_ptr<HloSharding> GetOpSharding(int op_idx) {
      return operand_shardings_[op_idx];
    };
    int64_t NumOpShardings() { return operand_shardings_.size(); }

    // modifying resulting sharding
    // TODO: accept a shared pointer
    void set_result_sharding(HloSharding result_sharding);

  private:
    // TODO: make these shared_ptr<const HloSharding>
    // The sharding of each operand of an instruction. Using shared_ptr
    // as noted by HloInstruction due to large size for many element tuples
    // This vector will be filled by enumerating incomplete sharding strategies
    std::vector<std::shared_ptr<HloSharding>> operand_shardings_;

    // TODO: make these shared_ptr<const HloSharding>
    // Sharding of result of computing instruction. This will be completed
    // by GSPMD when given the input shardings and determining the output
    // shardings.
    std::shared_ptr<HloSharding> result_sharding_;

    // Cost of this specific instruction sharding. This will be assigned
    // after evaluating the cost of the complete HloModule after performing
    // sharding propagation through SPMD.
    uint64_t cost_;

  };

  void InstructionSharding::AddOpSharding(HloSharding sharding) {
    operand_shardings_.push_back(std::make_shared<HloSharding>(sharding));
  }

  void InstructionSharding::set_result_sharding(HloSharding result_sharding) {
    result_sharding_ = std::make_shared<HloSharding>(result_sharding);
  }

  /*********************************************************/
  /* Sharding enumeration                                  */
  /*********************************************************/

  // enumerate sharding from the number of dimensions in the data
  // TODO: could be cached
  // Constructs a vector of rank * (rank + 1) shardings
  std::vector<HloSharding> EnumerateShardingsFromRank(int rank) {

    // two device dimensions currently (assume 4 (nodes) x 8 (gpus per node))
    std::vector<HloSharding> shardings;

    // note: this code is only acceptable for a 2D mesh grid,
    // would require more complicated solution for higher-dimensional grids
    for (int x_idx = 0; x_idx < rank; x_idx++) {
      for (int y_idx = 0; y_idx < rank; y_idx++) {
        // TODO: have a simple boolean for whether we would like to shard
        // both mesh grid dimensions on the same data dimension

        // construct tile_assignment_dims
        std::vector<int64_t> tile_assignment_dims(rank, 1);
        tile_assignment_dims[x_idx] *= MESH_X_DIM;
        tile_assignment_dims[y_idx] *= MESH_Y_DIM;

        // NOTE: intentionally may add two shardings if x_idx == y_idx
        // (i.e. when sharding a single data dimension on all devices)
        // because ordering of machines may influence resulting communication
        // costs and overall problem. Adding both shardings to be complete
        // construct the iota_reshape_dims and iota_tranpose_perm
        if (x_idx <= y_idx) {
          shardings.push_back(HloSharding::IotaTile(
            tile_assignment_dims,
            { MESH_X_DIM * MESH_Y_DIM },
            { 0 }
          ));
        }
        if (y_idx <= x_idx) {
          shardings.push_back(HloSharding::IotaTile(
            tile_assignment_dims,
            { MESH_X_DIM, MESH_Y_DIM },
            { 1, 0 }
          ));
        }
      }
    }

    return shardings;
  }

  // assuming a 2D mesh grid, enumerates all choice 2 shardings of data
  // TODO: determine if tuples of data will need to be considered for sharding
  std::vector<HloSharding> EnumerateGeneralOpSharding(HloInstruction* operand, 
      HloInstruction* instruction) {
    
    // operand requires sharding
    assert(operand->has_sharding());

    // only sharding array types of data, otherwise no sharding options
    const Shape op_shape = operand->shape();
    if (!op_shape.IsArray()) {
      return {};
    }

    return EnumerateShardingsFromRank(op_shape.rank());
  }

  // TODO: figure out a better way to deal with tuples for data
  std::vector<HloSharding> EnumerateTupleOpSharding(HloInstruction* operand,
      HloInstruction* instruction) {
    return {};
  }

  // Enumerates the shardings of a single operand instruction
  // depending on the user instruction of the operand and whether it is sharded.
  // This is a general function for iterating through shardings of a single
  // TODO: should give integer argument here and in EnumerateGeneralOpSharding
  std::vector<HloSharding> EnumerateOpSharding(
      HloInstruction* operand, HloInstruction* instruction) {
    
    // if sharding already exists for the instruction, only have that sharding
    if (operand->has_sharding()) {
      return { operand->sharding() };
    }

    // otherwise, perform sharding based on type of instruction
    // we are sharding operations for (may want to case on Dot product)
    switch (instruction->opcode()) {
    case HloOpcode::kTuple:
      return EnumerateTupleOpSharding(operand, instruction);
    default:
      return EnumerateGeneralOpSharding(operand, instruction);
    }

  }

  // Combine shardings for each operator to form sharding strategies
  std::vector<InstructionSharding> CombineShardingVectors(
      std::vector<std::vector<HloSharding>> sharding_vecs) {
    int num_vecs = sharding_vecs.size();

    if (num_vecs == 0) {
      return {};
    } else if (num_vecs == 1) {
      // only one operator, map each sharding to a separate InstructionSharding
      std::vector<InstructionSharding> strats;
      for (HloSharding sharding : sharding_vecs[0]) {
        InstructionSharding strat;
        strat.AddOpSharding(sharding);
        strats.push_back(strat);
      }
      return strats;
    }

    // otherwise recurse
    std::vector<HloSharding> shardings = sharding_vecs[num_vecs - 1];
    std::vector<InstructionSharding> sub_strats = CombineShardingVectors(
      std::vector<std::vector<HloSharding>>(sharding_vecs.begin(), 
        sharding_vecs.end() - 1)
    );

    std::vector<InstructionSharding> strats;
    for (HloSharding sharding : shardings) {
      for (InstructionSharding strat : sub_strats) {
        // copy the existing sub_strat and add the new sharding
        strat.AddOpSharding(sharding);
        strats.push_back(strat);
      }
    }
    
    return strats;
  }

  // Enumerates all possible sharding strategies on the inputs of the current
  // instruction
  // TODO: need to make instruction sharding use shared pointers
  // going to be many identical copies of the same sharding in memory
  // for larger problems
  std::vector<InstructionSharding> EnumerateInstructionShardings(
      HloInstruction* instruction) {

    // enumerate through the shardings for each operator of the instruction
    std::vector<std::vector<HloSharding>> all_op_shardings;

    // TODO: pass index of operand to distinguish from other operands
    // if necessary
    HloInstruction::InstructionVector operands = instruction->operands();
    for (HloInstruction* op : operands) {
      all_op_shardings.push_back(EnumerateOpSharding(op, instruction));
    }

    return CombineShardingVectors(all_op_shardings);
  } 

  /*********************************************************/
  /* GSPMD Completion                                      */
  /*********************************************************/

  // Major steps prior to evaluating the cost
  //  0. clone the original module?
  //  1. clear the module of shardings (does GSPMD insert any other metadata?)
  //  2. apply the shardings from a strategy
  //  3. run GSPMD
  //  4. evaluate the cost of the resulting module
  //  5. figure out the output sharding of the complete module

  // This function clears all shardings from instructions in the module
  void ClearHloShardings(HloModule* module) {

    for (HloComputation* computation : module->computations()) {
      for (HloInstruction* instruction : computation->instructions()) {
        instruction->clear_sharding();
      }
    }

    return;
  }

  // This function inserts a sharding strategy into an HloModule
  // Assumes that this is a single instruction HloModule
  void ApplyModuleStrategy(HloModule* module, InstructionSharding* strat) {

    std::shared_ptr<const HloSharding> sharding_ptr;

    // should only have one computation
    assert(module->computation_count() == 1);
    HloComputation* computation = module->entry_computation();

    // apply the shardings to the operands of the root instruction
    HloInstruction* root_instruction = computation->root_instruction();
    int num_operands = root_instruction->operand_count();
    assert(strat->NumOpShardings() == num_operands);

    for (int i = 0; i < num_operands; i++) {
      root_instruction->mutable_operand(i)->set_sharding(
        strat->GetOpSharding(i)
      );
    }

    return; 
  }

  // This function runs the sharding propagation pipeline pass on the module
  void RunGSPMD(HloModule* module) {

    // Setup HloPass for sharding propagation
    // Should not propagate sharding to parameters because parameters should
    // already have shardings
    HloPassPipeline spmd_pipeline("spmd-partitioner");
    spmd_pipeline.AddPass<ShardingPropagation>(
      /* is_spmd */ true,
      /* propagate_metadata */ false,
      /* sharding propagation to output */ absl::Span<const bool>({ true }),
      /* sharding propagation to parameters */ absl::Span<const bool>({ false })
    );

    // run pipeline
    spmd_pipeline.Run(module);

    return;
  }

  // This function returns the sharding of the entry computation's 
  // root instruction
  HloSharding GetRootSharding(HloModule* module) {
    HloInstruction* root = module->entry_computation()->root_instruction();
    assert(root->has_sharding());

    return root->sharding();
  }

  // This function will evaluate the sharding strategy on the 
  // single-instruction module by applying the input shardings from the strat
  // onto the operands of the module's root instruction, running GSPMD,
  // and evaluating the communication costs of the resulting module
  // The strat parameter will be updated with this cost and the resulting
  // output sharding
  void EvaluateShardingStrat(HloModule* module, InstructionSharding* strat) {

    // apply GSPMD to the module with the sharding strategy
    ClearHloShardings(module);
    ApplyModuleStrategy(module, strat);
    RunGSPMD(module);

    // now evaluate cost
    
    // update strat with cost and root instruction's output sharding

  }

  /*********************************************************/
  /* InstructionShardingInfo Class                         */
  /*********************************************************/

  class InstructionShardingInfo {
  public:
    InstructionShardingInfo(HloInstruction* orig_instr);
    ~InstructionShardingInfo() = default;

  private:

    // Applies the given sharding strategy to the module and perforsm GSPMD
    // on it to complete the sharding.

    // This function will iterate through the various sharding strategies
    // and apply them to the instruction within the module. Afterwards,
    // GSPMD will be run to complete the module and the appropriate 
    // sharding strategy
    void EstimateStrategyCosts();

    // Points to the original instruction that will have its
    // sharding strategies enumerated. Eventually, this instruction
    // will be modified with a sharding strategy provided by the solvers
    HloInstruction* orig_instr_;

    // Module containing a single computation of a single instruction that
    // is a clone of the orig_instr. Created on construction of class.
    // After enumerating various incomplete sharding strategies,
    // each incomplete strategy will be applied to this module, be completed
    // by GSPMD, and evaluated for it's computational and communication cost
    // by inspecting the completed module's resulting operations
    std::unique_ptr<HloModule> single_instr_module_;    

    // vector of sharding strategies for the given instruction
    std::vector<InstructionSharding> sharding_strats_;

  };

  InstructionShardingInfo::InstructionShardingInfo(HloInstruction* orig_instr) 
      : orig_instr_(orig_instr),
        single_instr_module_(CreateModuleFromInstruction(orig_instr)),
        sharding_strats_(EnumerateInstructionShardings(orig_instr)) {
    EstimateStrategyCosts();
    return;
  }

  void InstructionShardingInfo::EstimateStrategyCosts() {

    for (int i = 0; i < sharding_strats_.size(); i++) {
      EvaluateShardingStrat(single_instr_module_.get(), &sharding_strats_[i]);
    }

    return;
  }

}   // namespace


  /*********************************************************/
  /* AutoParallelizer Pass Implementation                  */
  /*********************************************************/

  // overriden functions from class
  // modifies the sharding specification of the instructions in the module
  // TODO: need to ignore the default modules that are not the core computation
  //  e.g. jit_convert_element_type, jit_broadcast_in_dim, jit__multi_slice
  // otherwise, just too much time spent on these things
  absl::StatusOr<bool> AutoParallelizer::Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) {

    VLOG(5) << "Testing AutoParallelizer Run";

    // create a clone of the module, then run off of that 
    std::unique_ptr<HloModule> module_clone = module->Clone();
    VLOG(5) << LOG_HEADER(0) << "module: " << module_clone->name();

    int num_computations = module_clone->computation_count();
    int comp_idx = 0;

    // iterate through HloModule computations
    for (HloComputation* computation : module_clone->computations()) {

      int num_instructions = computation->instruction_count();
      int instr_idx = 0;
      
      for (HloInstruction* instr : computation->instructions()) {

        // create the relevant sharding information for this instruction
        InstructionShardingInfo i(instr);

      }
    }

    VLOG(5) << "Done Testing AutoParallelizer Run";
    
    return true;
  }

    

    
}   // namespace xla