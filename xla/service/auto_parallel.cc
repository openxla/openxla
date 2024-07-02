// TODO: add copyright 

#include "xla/service/auto_parallel.h"

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tsl/platform/logging.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {

namespace {

  std::string LOG_HEADER(int x, const char c[]="AutoParallel: ") {
    return ((x == 0) ? (c) : ((LOG_HEADER(x - 1, c)) + "\t"));
  }

  /*********************************************************/
  /* Helper classes                                        */
  /*********************************************************/

  class ProgramShape;

  class InstructionShardingInfo {

  public:
    InstructionShardingInfo(HloInstruction* orig_instr);
    ~InstructionShardingInfo() = default;

  private:

    // Points to the original instruction that will have its
    // sharding strategies enumerated. Eventually, this instruction
    // will be modified with a sharding strategy provided by the solvers
    HloInstruction* orig_instr;

    // Module containing a single computation of a single instruction that
    // is a clone of the orig_instr. Created on construction of class.
    // After enumerating various incomplete sharding strategies,
    // each incomplete strategy will be applied to this module, be completed
    // by GSPMD, and evaluated for it's computational and communication cost
    // by inspecting the completed module's resulting operations
    HloModule* single_instr_module;    


  };



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

    return std::move(result); 
  }
  
  // Creates a module from a single instruction for running a simple pass on
  std::unique_ptr<HloModule> CreateModuleFromInstruction(HloInstruction* instruction) {

    // copy the instruction so as not to modify the HloModule
    std::unique_ptr<HloInstruction> instr_clone 
      = std::move(CloneSingleInstruction(instruction));
    
    // create entry computation from the single instruction
    HloComputation::Builder builder{"single-instr"};
    HloInstruction* instrp = builder.AddInstruction(std::move(instr_clone));
    std::unique_ptr<HloComputation> computation = builder.Build(instrp);

    // construct the module's configuration
    HloModuleConfig config{computation->ComputeProgramShape()};

    // construct the module from the computation (unique ptr so cleared out of memory)
    std::unique_ptr<HloModule> module = std::make_unique<HloModule>(std::string(instruction->name()), config);
    module->AddEntryComputation(std::move(computation));

    // create a copy so it is completely separate from original module
    std::unique_ptr<HloModule> module_clone = module->Clone(); 

    return module_clone;
  }

  /*********************************************************/
  /* Sharding enumeration                                  */
  /*********************************************************/

  // // creates a list of potential sharding strategies for the instruction
  // std::vector<std::unique_ptr<HloSharding>> GenerateShardingStrats(
  //     HloInstruction* instruction) {

  //   // figure things out here

  //   return nullptr; 
  // }

  void PrintInstructionInfo(HloInstruction* instruction, int depth=3) {

    int64_t num_operands = instruction->operand_count();
    VLOG(5) << LOG_HEADER(depth, "InstInfo: ") << "Name: " << instruction->name() << " " << instruction;
    VLOG(5) << LOG_HEADER(depth + 1, "InstInfo: ") << "num operands: " << num_operands;
    VLOG(5) << LOG_HEADER(depth + 1, "InstInfo: ") << "sharded: " << instruction->has_sharding();

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

}   // namespace


  /*********************************************************/
  /* AutoParallelizer Pass Implementation                  */
  /*********************************************************/

  // overriden functions from class
  // modifies the sharding specification of the instructions in the module
  absl::StatusOr<bool> AutoParallelizer::Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) {

    VLOG(5) << "Testing run for AutoParallelizer";

    PrintModuleInfo(module);

    // create a clone of the module, then run off of that 
    std::unique_ptr<HloModule> module_clone = module->Clone();
    VLOG(5) << LOG_HEADER(0) << "module: " << module_clone->name();


    PrintModuleInfo(module_clone.get());


    // iterate through HloModule computations
    for (HloComputation* computation : module_clone->computations()) {
      
      VLOG(5) << LOG_HEADER(1) << "computation: " << computation->name();

      for (HloInstruction* instr : computation->instructions()) {

        VLOG(5) << LOG_HEADER(2) << "instruction: " << instr->name() << " " << instr->shape().ToString() << " " << instr;
        // PrintInstructionInfo(instr);

        // regardless of whether it has sharding
        // include in problem but with fewer options
        std::unique_ptr<HloModule> single_inst_module = CreateModuleFromInstruction(instr);

        // now enumerate through various module shardings

      }
    }
    
    return true;
  }

    

    
}   // namespace xla