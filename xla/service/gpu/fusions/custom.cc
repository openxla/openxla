/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "xla/service/gpu/fusions/custom.h"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/ffi/attribute_map.h"
#include "xla/ffi/ffi_api.h"
#include "xla/hlo/analysis/while_loop_analysis.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/literal.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/fusions/fusion_emitter.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/kernel_arguments.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/service/gpu/kernels/custom_kernel_fusion.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/runtime/copy_thunk.h"
#include "xla/service/gpu/runtime/custom_call_target.h"
#include "xla/service/gpu/runtime/custom_call_thunk.h"
#include "xla/service/gpu/runtime/dynamic_slice_thunk.h"
#include "xla/service/gpu/runtime/gemm_thunk.h"
#include "xla/service/gpu/runtime/kernel_thunk.h"
#include "xla/service/gpu/runtime/nccl_all_reduce_thunk.h"
#include "xla/service/gpu/runtime/nccl_api.h"
#include "xla/service/gpu/runtime/nccl_collective_thunk.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/stream.h"
#include "xla/tools/hlo_extractor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

constexpr unsigned kGEMMOutputBufferIndex = 0;
constexpr unsigned kGEMMWorkspaceBufferIndex = 1;

// Gets the innermost while op surrounding the operation `op` if it exists. This
// also returns nullopt if there are multiple callers for the same while body.
std::optional<HloInstruction*> GetWhileOp(const HloInstruction& op,
                                          CallGraph& call_graph) {
  const HloInstruction* op_ptr = &op;
  while (op_ptr) {
    const HloComputation* parent = op_ptr->parent();
    std::vector<HloInstruction*> callers =
        call_graph.GetComputationCallers(parent);
    if (callers.size() != 1) {
      return std::nullopt;
    }
    if (callers[0]->opcode() == HloOpcode::kWhile) {
      return callers[0];
    }
    op_ptr = callers[0];
  }
  return std::nullopt;
}

// Returns true if the module has a `partition-id` or a `replica-id` operation,
// false otherwise.
bool DependsOnPartitionIdOrReplicaId(const HloModule& module) {
  for (const HloComputation* computation : module.computations()) {
    if (absl::c_any_of(computation->instructions(),
                       [](const HloInstruction* instr) -> bool {
                         return instr->opcode() == HloOpcode::kPartitionId ||
                                instr->opcode() == HloOpcode::kReplicaId;
                       })) {
      return true;
    }
  }
  return false;
}

// Extracts the offset module as a function of the induction variable. If unable
// to extract such a module, it returns nullptr.
std::unique_ptr<HloModule> ExtractAsFunctionOfIndvar(
    const HloInstruction& value, CallGraph& call_graph) {
  std::optional<HloInstruction*> while_op = GetWhileOp(value, call_graph);
  if (!while_op.has_value()) {
    VLOG(2) << "Unable to find surrounding while op" << value.ToString();
    return nullptr;
  }
  CHECK(while_op.value() != nullptr)
      << "GetWhileOp is not expected to return a nullptr.";
  const HloComputation* while_body = (*while_op)->while_body();
  std::optional<int64_t> indvar_idx = GetLoopInductionVarTupleIdx(*while_op);
  if (!indvar_idx.has_value()) {
    VLOG(2) << "Unable to find induction variable index for while op: "
            << (*while_op)->ToString();
    return nullptr;
  }
  if (while_body->num_parameters() != 1) {
    VLOG(2) << "Expected single parameter while body: "
            << while_body->ToString();
    return nullptr;
  }
  HloInstruction* indvar = hlo_query::GetUniqueGteInstruction(
      while_body->parameter_instruction(0), *indvar_idx);
  if (indvar == nullptr) {
    VLOG(2) << "Unable to find induction variable(idx=" << *indvar_idx
            << ") in while body: " << while_body->ToString();
    return nullptr;
  }
  std::unique_ptr<HloModule> value_module = ExtractModule(
      /*instruction=*/&value, /*height=*/-1,
      /*extract_selector=*/
      [indvar](const HloInstruction* instr) -> bool { return instr != indvar; },
      /*replace_type_selector=*/
      [](const HloInstruction*) -> ReplaceType {
        return ReplaceType::kReplaceParam;
      },
      /*cross_computation=*/true, /*inline_calls_and_fusions=*/true,
      /*run_verifier=*/true);
  if (value_module == nullptr) {
    VLOG(2) << "Unable to get value module for value: " << value.ToString();
    return nullptr;
  }
  if (DependsOnPartitionIdOrReplicaId(*value_module)) {
    VLOG(2) << "Value module depends on partition-id or replica-id: "
            << value_module->ToString();
    return nullptr;
  }
  HloComputation* entry = value_module->entry_computation();
  if (entry->num_parameters() != 1 ||
      entry->parameter_instruction(0)->shape() != indvar->shape()) {
    VLOG(2) << "Value module expected to depend on induction variable ("
            << indvar->shape().ToString() << ").";
    return nullptr;
  }
  return value_module;
}

// Extracts induction initialization module, making sure that it has zero
// parameters. If unable to extract such a module, it returns nullptr.
std::unique_ptr<HloModule> ExtractWhileInitModule(
    const HloInstruction* while_op) {
  std::optional<int64_t> indvar_idx = GetLoopInductionVarTupleIdx(while_op);
  if (!indvar_idx.has_value()) {
    VLOG(2) << "Unable to find induction variable index for while op: "
            << while_op->ToString();
    return nullptr;
  }
  const HloInstruction* init_tuple = while_op->operand(0);
  if (init_tuple->opcode() != HloOpcode::kTuple) {
    VLOG(2) << "Expected init to be a tuple, found: " << init_tuple->ToString();
    return nullptr;
  }
  const HloInstruction* init = init_tuple->operand(*indvar_idx);
  std::unique_ptr<HloModule> init_module = ExtractModule(/*instruction=*/init);
  if (init_module == nullptr) {
    VLOG(2) << "Unable to get init module for init: " << init->ToString();
    return nullptr;
  }
  if (DependsOnPartitionIdOrReplicaId(*init_module)) {
    VLOG(2) << "While loop init module depends on partition-id or replica-id: "
            << init_module->ToString();
    return nullptr;
  }
  if (init_module->entry_computation()->num_parameters() != 0) {
    VLOG(2) << "Expected no parameters for while loop init module, found: "
            << init_module->ToString();
    return nullptr;
  }
  return init_module;
}

// Extracts induction update module as a function of induction variable.
std::unique_ptr<HloModule> ExtractWhileUpdateModule(
    const HloInstruction* while_op, CallGraph& call_graph) {
  std::optional<int64_t> indvar_idx = GetLoopInductionVarTupleIdx(while_op);
  if (!indvar_idx.has_value()) {
    VLOG(2) << "Unable to find induction variable index for while op: "
            << while_op->ToString();
    return nullptr;
  }
  const HloInstruction* update_op =
      while_op->while_body()->root_instruction()->operand(*indvar_idx);
  std::unique_ptr<HloModule> update_module =
      ExtractAsFunctionOfIndvar(*update_op, call_graph);
  if (update_module == nullptr) {
    VLOG(2) << "Unable to get update module for update op: "
            << update_op->ToString();
    return nullptr;
  }
  // The conditions are checked as a part of the ExtractAsFunctionOfIndvar call.
  return update_module;
}

absl::StatusOr<std::unique_ptr<Thunk>> BuildCustomKernelThunkForFusion(
    IrEmitterContext& ir_emitter_context, const HloFusionInstruction& fusion,
    CustomKernel custom_kernel) {
  TF_ASSIGN_OR_RETURN(
      auto kernel_arguments,
      KernelArguments::Create(ir_emitter_context.buffer_assignment(), &fusion));

  return std::make_unique<CustomKernelThunk>(
      &fusion, std::move(custom_kernel), std::move(kernel_arguments.args()));
}

absl::StatusOr<BufferAllocation::Slice> GetOperandSlice(
    const BufferAssignment& buffer_assignment, const HloFusionAdaptor& adaptor,
    const HloInstruction& fusion_instr, const HloInstruction& start_instr,
    std::vector<HloInstruction*>& slice_instrs, const ShapeIndex& shape_idx,
    unsigned arg_idx) {
  if (const auto* param = DynCast<HloParameterInstruction>(&start_instr)) {
    return GetAllocationSlice(buffer_assignment,
                              fusion_instr.operand(param->parameter_number()),
                              shape_idx);
  }

  // Walk through ShapeIndex to find the real starting point.
  auto* start = const_cast<HloInstruction*>(&start_instr);
  for (auto idx : shape_idx) {
    CHECK(start->shape().IsTuple());
    start = const_cast<HloInstruction*>(start->operand(idx));
  }

  if (const auto* param = DynCast<HloParameterInstruction>(start)) {
    // At this point we've walked through all `shape_idx`, `index` should be
    // empty.
    return GetAllocationSlice(buffer_assignment,
                              fusion_instr.operand(param->parameter_number()),
                              /*index*/ {});
  }

  auto slice_adaptor = HloBfsFindIf(
      {HloInstructionAdaptor(*start, &adaptor)}, adaptor,
      [](HloInstructionAdaptor node) {
        return IsOpcodeAnyOf<HloOpcode::kDynamicSlice, HloOpcode::kSlice>(
            &node.instruction());
      });
  if (slice_adaptor.has_value()) {
    auto* slice_instr =
        const_cast<HloInstruction*>(&slice_adaptor->instruction());

    if (!IsContiguousSlice(*slice_instr)) {
      return absl::InternalError(
          "DynamicSliceFusion only handles contiguous slices "
          "currently");
    }

    slice_instrs[arg_idx] = slice_instr;

    const auto* param = Cast<HloParameterInstruction>(slice_instr->operand(0));
    // At this point we've walked through all `shape_idx`, `index` should be
    // empty.
    TF_ASSIGN_OR_RETURN(
        BufferAllocation::Slice orig_slice,
        GetAllocationSlice(buffer_assignment,
                           fusion_instr.operand(param->parameter_number()),
                           /*index*/ {}));

    if (auto* static_slice = DynCast<HloSliceInstruction>(slice_instr)) {
      // Update static slices.
      const Shape& src_shape = static_slice->operand(0)->shape();
      const Shape& dst_shape = static_slice->shape();
      int64_t size = ShapeUtil::ByteSizeOf(dst_shape);

      // Given this slice
      // f16[1,4,8]{2,1,0} slice(f16[2,8,8]{2,1,0}),
      //                         slice={[1:2], [4:8], [0:8]}
      //
      // The offset of the slice should be:
      //    slice_starts(0) * 8 * 8 * sizeof(f16) +
      //    slice_starts(1) * 8 * sizeof(f16)
      int64_t offset = orig_slice.offset();
      for (auto [start, stride] :
           llvm::zip(static_slice->slice_starts(),
                     *ShapeUtil::ByteStrides(src_shape))) {
        offset += start * stride;
      }

      return BufferAllocation::Slice(orig_slice.allocation(), offset, size);
    }

    return orig_slice;
  }

  return absl::InternalError("WTF");
}

absl::Status CollectSliceInfo(
    const BufferAssignment& buffer_assignment,
    const HloInstruction& fusion_instr,
    absl::Span<HloInstruction*> slice_instrs,
    std::vector<std::optional<std::vector<DynamicSliceThunk::Offset>>>& offsets,
    std::vector<std::optional<Shape>>& orig_shapes,
    std::vector<std::optional<Shape>>& sliced_shapes,
    std::vector<std::optional<uint64_t>>& offset_byte_sizes, unsigned arg_idx,
    bool can_compute_indvar_on_host,
    std::vector<std::unique_ptr<HloModule>>& fake_modules,
    CallGraph& call_graph) {
  auto* arg_slice_instr =
      DynCastOrNull<HloDynamicIndexInstruction>(slice_instrs[arg_idx]);
  if (arg_slice_instr == nullptr) {
    return absl::OkStatus();
  }

  std::vector<DynamicSliceThunk::Offset> arg_offsets;
  for (auto idx_op : arg_slice_instr->index_operands()) {
    const auto* param = Cast<HloParameterInstruction>(idx_op);
    const auto* offset_value = fusion_instr.operand(param->parameter_number());

    // Try to evaluate the offset value, maybe it is simple arithmetic.
    absl::StatusOr<Literal> offset_literal = HloEvaluator().Evaluate(
        /*instruction=*/offset_value,
        /*precomputed_analyses=*/{},
        /*recursively_evaluate_nonconstant_operands=*/true);

    if (offset_literal.ok()) {
      std::optional<int64_t> offset_value =
          LiteralUtil::LiteralAsScalarInt64(offset_literal.value());
      if (!offset_value.has_value()) {
        return absl::InternalError(
            absl::StrCat("Unsupported constant offset shape: ",
                         offset_literal->shape().ToString()));
      }
      arg_offsets.emplace_back() = *offset_value;
    } else if (std::unique_ptr<HloModule> offset_module =
                   ExtractAsFunctionOfIndvar(*offset_value, call_graph);
               offset_module && can_compute_indvar_on_host) {
      fake_modules.push_back(std::move(offset_module));
      arg_offsets.emplace_back() = fake_modules.back().get();
    } else {
      // Loop offset computed on device and has to be transferred to host.
      TF_ASSIGN_OR_RETURN(arg_offsets.emplace_back(),
                          GetAllocationSlice(buffer_assignment, offset_value,
                                             /*index=*/{}));
    }
  }
  offsets[arg_idx] = std::move(arg_offsets);
  orig_shapes[arg_idx] = arg_slice_instr->operand(0)->shape();
  sliced_shapes[arg_idx] = DynCast<HloDynamicSliceInstruction>(arg_slice_instr)
                               ? arg_slice_instr->shape()
                               : arg_slice_instr->operand(1)->shape();
  offset_byte_sizes[arg_idx] = ShapeUtil::ByteSizeOfPrimitiveType(
      arg_slice_instr->index_operands().front()->shape().element_type());

  return absl::OkStatus();
}

// This function assumes that the computation graph for `fusion_instr` looks
// like:
//
//   ...
//   root_tuple_operand = (... ty[shape], ...) ...
//   ROOT root_tuple = (... (... ty[shape], ...), ...)
//     tuple(... root_tuple_operand, ...)
//
// Given such a pattern and a (complete) index into `root_tuple_operand`, we
// recover the slice of `root_tuple` that corresponds to that index.
absl::StatusOr<BufferAllocation::Slice> GetResultSliceForPartiallyUnnestedTuple(
    const BufferAssignment& buffer_assignment,
    const HloFusionInstruction& fusion_instr,
    const HloInstruction& root_tuple_operand,
    const ShapeIndex& root_tuple_operand_shape_idx,
    const HloInstruction& root_tuple) {
  int64_t operand_index = root_tuple.operand_index(&root_tuple_operand);
  ShapeIndex slice_shape_index;
  slice_shape_index.push_back(operand_index);
  absl::c_copy(root_tuple_operand_shape_idx,
               std::back_inserter(slice_shape_index));
  return GetAllocationSlice(buffer_assignment, &fusion_instr,
                            slice_shape_index);
}

absl::StatusOr<BufferAllocation::Slice> GetResultSlice(
    const BufferAssignment& buffer_assignment, const HloFusionAdaptor& adaptor,
    const HloFusionInstruction& fusion_instr, const HloInstruction& start_instr,
    std::vector<HloInstruction*>& slice_instrs, const ShapeIndex& shape_idx,
    unsigned arg_idx) {
  auto* start = const_cast<HloInstruction*>(&start_instr);
  if (start->IsRoot()) {
    return GetAllocationSlice(buffer_assignment, &fusion_instr, shape_idx);
  }

  // Walk through ShapeIndex to find the real "user" (i.e. not get-tuple-element
  // user). Otherwise one sliced element will mark all buffers of all other
  // elements "sliced" too.
  if (start->shape().IsTuple()) {
    for (auto [index_nesting_level, index_in_shape] :
         llvm::enumerate(shape_idx)) {
      HloInstruction* gte_user = nullptr;
      for (auto* user : start->users()) {
        if (user->opcode() == HloOpcode::kGetTupleElement &&
            user->tuple_index() == index_in_shape) {
          gte_user = user;
          break;
        }
      }

      if (gte_user == nullptr) {
        // At this point, two things are known:
        //   1. `start` was not the root instruction of the fusion at the
        //      beginning of this function call;
        //   2. `start` still has a tuple shape because we haven't managed to
        //      unwrap the entire shape index.
        // We also know, by definition of the surrounding pass, that all the
        // results of the custom call must be materialized at the output of
        // the fusion, which indicates that `start` is currently *not* the
        // root. Since we can't slice/bitcast/reshape a tuple, then the
        // only possible consumer should be a `tuple` instruction, which
        // logically should be the root of the fusion.
        HloInstruction* start_user = start->users().front();
        if (start->user_count() != 1 ||
            start_user->opcode() != HloOpcode::kTuple ||
            !start_user->IsRoot()) {
          return absl::InternalError(
              "Expected the user of a nested tuple shape to be a root tuple "
              "instruction."
              "Expected a single user of the tuple-shaped instruction");
        }

        ShapeIndex remaining_shape_index(
            shape_idx.begin() + index_nesting_level, shape_idx.end());
        return GetResultSliceForPartiallyUnnestedTuple(
            buffer_assignment, fusion_instr, *start, remaining_shape_index,
            *start_user);
      }

      start = gte_user;
    }
  }

  auto slice_adaptor = HloBfsFindIf(
      {HloInstructionAdaptor(*start, &adaptor)}, adaptor,
      [](auto node) { return node.opcode() == HloOpcode::kDynamicUpdateSlice; },
      /*visit_operands=*/false);
  if (slice_adaptor.has_value()) {
    auto* slice_instr =
        const_cast<HloInstruction*>(&slice_adaptor->instruction());
    slice_instrs[arg_idx] = slice_instr;

    if (!IsContiguousSlice(*slice_instr)) {
      return absl::InternalError(
          "DynamicSliceFusion only handles contiguous slices "
          "currently");
    }

    // If slice is the root of the fusion, get the buffer assigned to the
    // fusion itself.
    if (slice_instr == fusion_instr.fused_expression_root()) {
      return GetAllocationSlice(buffer_assignment, &fusion_instr, {});
    }
  }

  constexpr absl::string_view kNonContiguousDynamicUpdateSliceError =
      "DynamicSliceFusion only handles contiguous slices currently";

  // At this point, we've fully unfolded a tuple that was not the root of the
  // computation. There are two options; either, the root is a tuple, or it is
  // not.
  //
  // If the root is not a tuple, we can simply get the buffer slice assigned to
  // the fusion itself---there is nothing else to choose from.
  if (fusion_instr.shape().IsArray()) {
    HloInstruction* root = fusion_instr.fused_expression_root();
    if (root->opcode() == HloOpcode::kDynamicUpdateSlice &&
        !IsContiguousSlice(*root)) {
      return absl::InternalError(kNonContiguousDynamicUpdateSliceError);
    }
    return GetAllocationSlice(buffer_assignment, &fusion_instr, {});
  }

  // If the root is a tuple however, it may be a nested tuple. Go all the way
  // to the root to figure out the index that our array occupies within that
  // tuple.
  HloInstruction* current_hlo = start;
  std::vector<int64_t> reversed_shape_index;
  do {
    TF_RET_CHECK(current_hlo->user_count() == 1);
    HloInstruction* user = current_hlo->users().front();
    // We may encounter three ops here: dynamic-update-slice, tuple, or bitcast.
    switch (user->opcode()) {
      case HloOpcode::kBitcast:
        break;
      case HloOpcode::kDynamicUpdateSlice:
        if (!IsContiguousSlice(*user)) {
          return absl::InternalError(kNonContiguousDynamicUpdateSliceError);
        }
        break;
      case HloOpcode::kTuple:
        reversed_shape_index.push_back(user->operand_index(current_hlo));
        break;
      default:
        return absl::InternalError(
            absl::StrCat("Unexpected opcode while processing the epilogue of a "
                         "DynamicSliceFusion: ",
                         HloOpcodeString(user->opcode())));
    };
    current_hlo = user;
  } while (!current_hlo->IsRoot());

  return GetAllocationSlice(
      buffer_assignment, &fusion_instr,
      ShapeIndex(reversed_shape_index.rbegin(), reversed_shape_index.rend()));
}

absl::StatusOr<FusionEmissionResult> EmitGemm(
    IrEmitterContext& ir_emitter_context, const HloFusionAdaptor& adaptor,
    const HloFusionInstruction& fusion,
    const HloCustomCallInstruction& custom_call, CallGraph& call_graph) {
  const BufferAssignment& buffer_assignment =
      ir_emitter_context.buffer_assignment();

  std::vector<std::optional<std::vector<DynamicSliceThunk::Offset>>>
      offset_buffer_indices(4, std::nullopt);
  std::vector<std::optional<Shape>> orig_shapes(4, std::nullopt);
  std::vector<std::optional<Shape>> sliced_shapes(4, std::nullopt);
  std::vector<std::optional<uint64_t>> offset_byte_sizes(4, std::nullopt);

  std::vector<HloInstruction*> slice_instrs(4, nullptr);

  std::vector<std::unique_ptr<HloModule>> fake_modules;

  // The flag `can_compute_indvar_on_host` tells whether the value of induction
  // variable can be calculated at runtime on the host. This is useful to
  // determine whether to generate an offset module or not.
  std::optional<HloInstruction*> while_op = GetWhileOp(fusion, call_graph);
  std::unique_ptr<HloModule> init_module, update_module;
  if (while_op != std::nullopt) {
    CHECK(while_op.value() != nullptr)
        << "GetWhileOp is not expected to return nullptr.";
    init_module = ExtractWhileInitModule(*while_op);
    update_module = ExtractWhileUpdateModule(*while_op, call_graph);
  }
  bool can_compute_indvar_on_host =
      (init_module != nullptr && update_module != nullptr);

  unsigned arg_idx = 0;
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice lhs_slice,
                      GetOperandSlice(buffer_assignment, adaptor, fusion,
                                      *custom_call.operand(arg_idx),
                                      slice_instrs, /*shape_idx=*/{}, arg_idx));
  TF_RETURN_IF_ERROR(CollectSliceInfo(
      buffer_assignment, fusion, absl::Span<HloInstruction*>(slice_instrs),
      offset_buffer_indices, orig_shapes, sliced_shapes, offset_byte_sizes,
      arg_idx++, can_compute_indvar_on_host, fake_modules, call_graph));

  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice rhs_slice,
                      GetOperandSlice(buffer_assignment, adaptor, fusion,
                                      *custom_call.operand(arg_idx),
                                      slice_instrs, /*shape_idx=*/{}, arg_idx));
  TF_RETURN_IF_ERROR(CollectSliceInfo(
      buffer_assignment, fusion, absl::Span<HloInstruction*>(slice_instrs),
      offset_buffer_indices, orig_shapes, sliced_shapes, offset_byte_sizes,
      arg_idx++, can_compute_indvar_on_host, fake_modules, call_graph));

  BufferAllocation::Slice output;
  std::optional<BufferAllocation::Slice> workspace = std::nullopt;
  std::optional<BufferAllocation::Slice> slice_workspace_fake = std::nullopt;

  // Handling cases where multiple operands share the same buffer, with
  // different offset by creating new fake allocations so each operand will have
  // a different buffer index. The slices can thus always start at offset 0.
  // DynamicSliceThunk will take care of the offset adjustment.
  std::vector<std::unique_ptr<BufferAllocation>> fake_allocations(4);
  if (fusion.shape().IsArray()) {
    TF_ASSIGN_OR_RETURN(
        output, GetResultSlice(buffer_assignment, adaptor, fusion, custom_call,
                               slice_instrs, /*shape_idx=*/{}, arg_idx));
    TF_RETURN_IF_ERROR(CollectSliceInfo(
        buffer_assignment, fusion, absl::Span<HloInstruction*>(slice_instrs),
        offset_buffer_indices, orig_shapes, sliced_shapes, offset_byte_sizes,
        arg_idx, can_compute_indvar_on_host, fake_modules, call_graph));
  } else {
    TF_ASSIGN_OR_RETURN(
        output,
        GetResultSlice(buffer_assignment, adaptor, fusion, custom_call,
                       slice_instrs, /*shape_idx=*/{kGEMMOutputBufferIndex},
                       arg_idx));
    TF_RETURN_IF_ERROR(CollectSliceInfo(
        buffer_assignment, fusion, absl::Span<HloInstruction*>(slice_instrs),
        offset_buffer_indices, orig_shapes, sliced_shapes, offset_byte_sizes,
        arg_idx++, can_compute_indvar_on_host, fake_modules, call_graph));

    // TODO(vuson): If we want to support slices of workspace, we'd need to
    // start `HloFindIf` with `get-tuple-element` with the right index.
    TF_ASSIGN_OR_RETURN(
        workspace, GetAllocationSlice(buffer_assignment, &fusion,
                                      /*index=*/{kGEMMWorkspaceBufferIndex}));
    TF_RETURN_IF_ERROR(CollectSliceInfo(
        buffer_assignment, fusion, absl::Span<HloInstruction*>(slice_instrs),
        offset_buffer_indices, orig_shapes, sliced_shapes, offset_byte_sizes,
        arg_idx, can_compute_indvar_on_host, fake_modules, call_graph));
    fake_allocations[arg_idx] = std::make_unique<BufferAllocation>(
        /*index=*/arg_idx, workspace->size(), /*color=*/0);
    slice_workspace_fake = BufferAllocation::Slice(
        fake_allocations[arg_idx].get(), 0, workspace->size());
  }

  if (absl::c_all_of(slice_instrs, [&](auto slice_instr) {
        return slice_instr == nullptr;
      })) {
    return absl::InternalError(
        "DynamicSliceFusion expects at least one sliced "
        "operand/result");
  }

  const bool deterministic_ops =
      RequireDeterminism(fusion.GetModule()->config());

  TF_ASSIGN_OR_RETURN(
      GemmConfig config,
      GemmConfig::For(static_cast<const HloInstruction*>(&custom_call)));

  std::unique_ptr<Thunk> thunk;
  auto thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(&fusion);

  if (absl::c_any_of(slice_instrs, [&](auto slice_instr) {
        return DynCastOrNull<HloDynamicIndexInstruction>(slice_instr) !=
               nullptr;
      })) {
    // Creating embedded GEMM thunk.
    unsigned fake_arg_idx = 0;
    int64_t lhs_byte_size =
        ShapeUtil::ByteSizeOf(custom_call.operand(fake_arg_idx)->shape());
    fake_allocations[fake_arg_idx] = std::make_unique<BufferAllocation>(
        /*index=*/fake_arg_idx, lhs_byte_size, /*color=*/0);
    BufferAllocation::Slice slice_lhs_fake(fake_allocations[fake_arg_idx].get(),
                                           0, lhs_byte_size);

    fake_arg_idx++;
    int64_t rhs_byte_size =
        ShapeUtil::ByteSizeOf(custom_call.operand(fake_arg_idx)->shape());
    fake_allocations[fake_arg_idx] = std::make_unique<BufferAllocation>(
        /*index=*/fake_arg_idx, rhs_byte_size, /*color=*/0);
    BufferAllocation::Slice slice_rhs_fake(fake_allocations[fake_arg_idx].get(),
                                           0, rhs_byte_size);

    fake_arg_idx++;
    int64_t out_fake_byte_size = ShapeUtil::ByteSizeOf(
        custom_call.shape().IsArray() ? custom_call.shape()
                                      : custom_call.shape().tuple_shapes(0));
    fake_allocations[fake_arg_idx] = std::make_unique<BufferAllocation>(
        /*index=*/fake_arg_idx, out_fake_byte_size, /*color=*/0);
    BufferAllocation::Slice slice_out_fake(fake_allocations[fake_arg_idx].get(),
                                           0, out_fake_byte_size);
    ThunkSequence seq;
    seq.emplace_back(std::make_unique<GemmThunk>(
        thunk_info, std::move(config), slice_lhs_fake, slice_rhs_fake,
        slice_out_fake, slice_workspace_fake, deterministic_ops));

    std::vector<std::optional<BufferAllocation::Slice>> arguments{
        lhs_slice, rhs_slice, output, workspace};

    thunk = std::make_unique<DynamicSliceThunk>(
        thunk_info, std::make_unique<ThunkSequence>(std::move(seq)),
        std::move(arguments), std::move(fake_allocations),
        std::move(offset_buffer_indices), std::move(orig_shapes),
        std::move(sliced_shapes), std::move(offset_byte_sizes),
        std::move(fake_modules), std::move(init_module),
        std::move(update_module));
  } else {
    thunk = std::make_unique<GemmThunk>(thunk_info, std::move(config),
                                        lhs_slice, rhs_slice, output, workspace,
                                        deterministic_ops);
  }

  FusionEmissionResult result;
  result.thunks.push_back(std::move(thunk));
  return result;
}

absl::StatusOr<FusionEmissionResult> EmitCustomCall(
    IrEmitterContext& ir_emitter_context, const HloFusionAdaptor& adaptor,
    const HloFusionInstruction& fusion,
    const HloCustomCallInstruction& custom_call, CallGraph& call_graph) {
  const BufferAssignment& buffer_assignment =
      ir_emitter_context.buffer_assignment();

  const std::string& call_target_name = custom_call.custom_call_target();

  // Typed FFI custom calls is a replacement for legacy custom calls with
  // a rich type safe API. It's under construction and not fully supported.
  bool is_ffi_custom_call =
      custom_call.api_version() == CustomCallApiVersion::API_VERSION_TYPED_FFI;

  void* call_target = CustomCallTargetRegistry::Global()->Lookup(
      call_target_name, std::string(ir_emitter_context.platform_name()));

  absl::StatusOr<ffi::HandlerRegistration> registration =
      ffi::FindHandler(call_target_name, ir_emitter_context.platform_name());

  // At least one implementation should be available at run time.
  bool found_custom_call = !is_ffi_custom_call && call_target != nullptr;
  bool found_ffi_handler = is_ffi_custom_call && registration.ok();

  if (!found_custom_call && !found_ffi_handler) {
    return absl::InternalError(
        "DynamicSliceFusion expects custom calls that are emittable as "
        "thunks");
  }

  using Slices = std::vector<std::optional<CustomCallThunk::Slice>>;

  int64_t num_args = ShapeUtil::GetLeafCount(custom_call.shape());
  absl::c_for_each(custom_call.operands(), [&](auto* operand) {
    num_args += ShapeUtil::GetLeafCount(operand->shape());
  });

  std::vector<std::optional<std::vector<DynamicSliceThunk::Offset>>> offsets(
      num_args, std::nullopt);
  std::vector<std::optional<Shape>> orig_shapes(num_args, std::nullopt);
  std::vector<std::optional<Shape>> sliced_shapes(num_args, std::nullopt);
  std::vector<std::optional<uint64_t>> offset_byte_sizes(num_args,
                                                         std::nullopt);

  std::vector<HloInstruction*> slice_instrs(num_args, nullptr);
  std::vector<std::optional<BufferAllocation::Slice>> arguments;

  std::vector<std::unique_ptr<HloModule>> fake_modules;

  // The flag `can_compute_indvar_on_host` tells whether the value of induction
  // variable can be calculated at runtime on the host. This is useful to
  // determine whether to generate an offset module or not.
  std::optional<HloInstruction*> while_op = GetWhileOp(fusion, call_graph);
  std::unique_ptr<HloModule> init_module, update_module;
  if (while_op != std::nullopt) {
    CHECK(while_op.value() != nullptr)
        << "GetWhileOp is not expected to return nullptr.";
    init_module = ExtractWhileInitModule(*while_op);
    update_module = ExtractWhileUpdateModule(*while_op, call_graph);
  }
  bool can_compute_indvar_on_host =
      (init_module != nullptr && update_module != nullptr);

  unsigned arg_idx = 0;
  // TODO(vuson): add test for custom call with token-typed operands
  Slices operands;
  for (auto* operand : custom_call.operands()) {
    TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
        operand->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
          if (subshape.IsToken()) {
            arg_idx++;
            operands.push_back(std::nullopt);
            return absl::OkStatus();
          }
          if (!subshape.IsArray()) {
            return absl::OkStatus();
          }
          TF_ASSIGN_OR_RETURN(
              auto slice,
              GetOperandSlice(buffer_assignment, adaptor, fusion, *operand,
                              slice_instrs, /*shape_idx=*/index, arg_idx));
          TF_RETURN_IF_ERROR(CollectSliceInfo(
              buffer_assignment, fusion,
              absl::Span<HloInstruction*>(slice_instrs), offsets, orig_shapes,
              sliced_shapes, offset_byte_sizes, arg_idx++,
              can_compute_indvar_on_host, fake_modules, call_graph));

          operands.push_back(CustomCallThunk::Slice{slice, subshape});
          arguments.push_back(slice);
          return absl::OkStatus();
        }));
  }

  Slices results;
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      custom_call.shape(), [&](const Shape& subshape, const ShapeIndex& index) {
        if (subshape.IsToken()) {
          arg_idx++;
          results.push_back(std::nullopt);
          return absl::OkStatus();
        }
        if (!subshape.IsArray()) {
          return absl::OkStatus();
        }
        TF_ASSIGN_OR_RETURN(
            auto slice,
            GetResultSlice(buffer_assignment, adaptor, fusion, custom_call,
                           slice_instrs, /*shape_idx=*/index, arg_idx));
        TF_RETURN_IF_ERROR(CollectSliceInfo(
            buffer_assignment, fusion,
            absl::Span<HloInstruction*>(slice_instrs), offsets, orig_shapes,
            sliced_shapes, offset_byte_sizes, arg_idx++,
            can_compute_indvar_on_host, fake_modules, call_graph));

        results.push_back(CustomCallThunk::Slice{slice, subshape});
        arguments.push_back(slice);
        return absl::OkStatus();
      }));

  if (absl::c_all_of(slice_instrs, [&](auto slice_instr) {
        return slice_instr == nullptr;
      })) {
    return absl::InternalError(
        "DynamicSliceFusion expects at least one sliced "
        "operand/result");
  }

  // For legacy custom calls we convert all API versions into the latest
  // status-returning one and pass backend config as an opaque string.
  CustomCallThunk::CustomCallTarget custom_call_target;

  // For XLA FFI handlers we decode opaque backend config into attributes map
  // at IR emission time, so that we do not need to parse MLIR at run time. For
  // FFI handlers backend config must be a compatible MLIR dictionary.
  CustomCallThunk::AttributesMap attributes;

  // For information about this calling convention, see
  // xla/g3doc/custom_call.md.
  switch (custom_call.api_version()) {
    case CustomCallApiVersion::API_VERSION_ORIGINAL:
      custom_call_target = [call_target](se::Stream* stream, void** buffers,
                                         const char* opaque, size_t opaque_len,
                                         XlaCustomCallStatus*) {
        reinterpret_cast<CustomCallWithOpaqueStreamHandle>(call_target)(
            stream->platform_specific_handle().stream, buffers, opaque,
            opaque_len);
      };
      break;
    case CustomCallApiVersion::API_VERSION_STATUS_RETURNING:
    case CustomCallApiVersion::API_VERSION_STATUS_RETURNING_UNIFIED:
      custom_call_target = [call_target](se::Stream* stream, void** buffers,
                                         const char* opaque, size_t opaque_len,
                                         XlaCustomCallStatus* status) {
        reinterpret_cast<CustomCallWithStatusAndOpaqueStreamHandle>(
            call_target)(stream->platform_specific_handle().stream, buffers,
                         opaque, opaque_len, status);
      };
      break;
    case CustomCallApiVersion::API_VERSION_TYPED_FFI:
      // We already checked `handler` above.
      break;
    default:
      return Internal("Unknown custom-call API version enum value: %d",
                      custom_call.api_version());
  }

  auto backend_config = custom_call.backend_config<GpuBackendConfig>();
  if (!backend_config.ok()) {
    LOG(WARNING) << "Unable to parse backend config for custom call: "
                 << backend_config.status().message() << "\n"
                 << "Fall back to parse the raw backend config str.";
  }

  std::unique_ptr<Thunk> thunk;
  auto thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(&fusion);

  auto ffi_thunk =
      [&](Slices ops,
          Slices res) -> absl::StatusOr<std::unique_ptr<CustomCallThunk>> {
    auto& called_computations = custom_call.called_computations();
    auto& backend_config_str =
        backend_config.ok()
            ? backend_config->custom_call_backend_config().attributes()
            : custom_call.raw_backend_config_string();
    if (!backend_config_str.empty()) {
      mlir::Attribute attr = mlir::parseAttribute(
          backend_config_str, ir_emitter_context.mlir_context());
      auto dict = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(attr);
      if (dict == nullptr) {
        return absl::InternalError(
            "Unsupported backend config. Expected a string parsable into "
            "dictionary attribute");
      }
      TF_ASSIGN_OR_RETURN(attributes, xla::ffi::BuildAttributesMap(dict));
    }
    return CustomCallThunk::Create(
        thunk_info, call_target_name, registration->bundle, std::move(ops),
        std::move(res), std::move(attributes),
        called_computations.empty() ? nullptr : called_computations[0]);
  };

  auto legacy_thunk =
      [&](Slices ops,
          Slices res) -> absl::StatusOr<std::unique_ptr<CustomCallThunk>> {
    std::string opaque =
        backend_config.ok()
            ? backend_config->custom_call_backend_config().opaque()
            : custom_call.raw_backend_config_string();
    return CustomCallThunk::Create(
        thunk_info, call_target_name, std::move(custom_call_target),
        std::move(ops), std::move(res), std::move(opaque));
  };

  std::vector<std::unique_ptr<BufferAllocation>> fake_allocations(num_args);
  if (absl::c_any_of(slice_instrs, [&](auto slice_instr) {
        return DynCastOrNull<HloDynamicIndexInstruction>(slice_instr) !=
               nullptr;
      })) {
    // Creating embedded custom call thunk.
    unsigned fake_arg_idx = 0;

    Slices fake_operands;
    for (auto* operand : custom_call.operands()) {
      TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
          operand->shape(),
          [&](const Shape& subshape, const ShapeIndex& index) {
            if (subshape.IsToken()) {
              fake_arg_idx++;
              fake_operands.push_back(std::nullopt);
              return absl::OkStatus();
            }
            if (!subshape.IsArray()) {
              return absl::OkStatus();
            }

            int64_t operand_byte_size = ShapeUtil::ByteSizeOf(subshape);
            fake_allocations[fake_arg_idx] = std::make_unique<BufferAllocation>(
                /*index=*/fake_arg_idx, operand_byte_size, /*color=*/0);
            BufferAllocation::Slice fake_slice(
                fake_allocations[fake_arg_idx].get(), 0, operand_byte_size);

            fake_arg_idx++;
            fake_operands.push_back(
                CustomCallThunk::Slice{fake_slice, subshape});
            return absl::OkStatus();
          }));
    }

    Slices fake_results;
    TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
        custom_call.shape(),
        [&](const Shape& subshape, const ShapeIndex& index) {
          if (subshape.IsToken()) {
            fake_arg_idx++;
            fake_results.push_back(std::nullopt);
            return absl::OkStatus();
          }
          if (!subshape.IsArray()) {
            return absl::OkStatus();
          }

          int64_t result_byte_size = ShapeUtil::ByteSizeOf(subshape);
          fake_allocations[fake_arg_idx] = std::make_unique<BufferAllocation>(
              /*index=*/fake_arg_idx, result_byte_size, /*color=*/0);
          BufferAllocation::Slice fake_slice(
              fake_allocations[fake_arg_idx].get(), 0, result_byte_size);

          fake_arg_idx++;
          fake_results.push_back(CustomCallThunk::Slice{fake_slice, subshape});
          return absl::OkStatus();
        }));

    ThunkSequence seq;
    TF_ASSIGN_OR_RETURN(
        seq.emplace_back(),
        found_ffi_handler
            ? ffi_thunk(std::move(fake_operands), std::move(fake_results))
            : legacy_thunk(std::move(fake_operands), std::move(fake_results)));

    thunk = std::make_unique<DynamicSliceThunk>(
        thunk_info, std::make_unique<ThunkSequence>(std::move(seq)),
        std::move(arguments), std::move(fake_allocations), std::move(offsets),
        std::move(orig_shapes), std::move(sliced_shapes),
        std::move(offset_byte_sizes), std::move(fake_modules),
        std::move(init_module), std::move(update_module));
  } else {
    TF_ASSIGN_OR_RETURN(
        thunk, found_ffi_handler
                   ? ffi_thunk(std::move(operands), std::move(results))
                   : legacy_thunk(std::move(operands), std::move(results)));
  }

  FusionEmissionResult result;
  result.thunks.push_back(std::move(thunk));
  return result;
}

template <typename NcclThunkType, typename HloInstType>
absl::StatusOr<FusionEmissionResult> EmitCollective(
    IrEmitterContext& ir_emitter_context, const HloFusionAdaptor& adaptor,
    const HloFusionInstruction& fusion_instr, const HloInstType* instr,
    bool use_global_device_ids, CallGraph& call_graph) {
  Thunk::Kind collective_done_thunk_kind;
  switch (instr->opcode()) {
    case HloOpcode::kReduceScatter:
      collective_done_thunk_kind = Thunk::kNcclReduceScatterDone;
      break;
    default:
      return absl::InternalError(
          "Unexpected operation in dynamic slice fusion");
  }

  const BufferAssignment& buffer_assignment =
      ir_emitter_context.buffer_assignment();

  std::vector<std::optional<std::vector<DynamicSliceThunk::Offset>>>
      offset_buffer_indices(2, std::nullopt);
  std::vector<std::optional<Shape>> orig_shapes(2, std::nullopt);
  std::vector<std::optional<Shape>> sliced_shapes(2, std::nullopt);
  std::vector<std::optional<uint64_t>> offset_byte_sizes(2, std::nullopt);

  std::vector<HloInstruction*> slice_instrs(2, nullptr);
  std::vector<std::optional<BufferAllocation::Slice>> arguments;

  std::vector<std::unique_ptr<HloModule>> fake_modules;

  // The flag `can_compute_indvar_on_host` tells whether the value of induction
  // variable can be calculated at runtime on the host. This is useful to
  // determine whether to generate an offset module or not.
  std::optional<HloInstruction*> while_op =
      GetWhileOp(fusion_instr, call_graph);
  std::unique_ptr<HloModule> init_module, update_module;
  if (while_op != std::nullopt) {
    CHECK(while_op.value() != nullptr)
        << "GetWhileOp is not expected to return nullptr.";
    init_module = ExtractWhileInitModule(*while_op);
    update_module = ExtractWhileUpdateModule(*while_op, call_graph);
  }
  bool can_compute_indvar_on_host =
      (init_module != nullptr && update_module != nullptr);

  // Collect slice information for inputs.
  unsigned arg_idx = 0;
  TF_ASSIGN_OR_RETURN(
      BufferAllocation::Slice src,
      GetOperandSlice(buffer_assignment, adaptor, fusion_instr,
                      /*start_instr=*/*instr->operand(arg_idx), slice_instrs,
                      /*shape_idx=*/{}, arg_idx));
  arguments.push_back(src);
  TF_RETURN_IF_ERROR(CollectSliceInfo(
      buffer_assignment, fusion_instr,
      /*slice_instrs=*/absl::Span<HloInstruction*>(slice_instrs),
      /*offsets=*/offset_buffer_indices, orig_shapes, sliced_shapes,
      offset_byte_sizes, arg_idx, can_compute_indvar_on_host, fake_modules,
      call_graph));
  arg_idx++;

  // Collect slice information for outputs.
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice dst,
                      GetResultSlice(buffer_assignment, adaptor, fusion_instr,
                                     /*start_instr=*/*instr, slice_instrs,
                                     /*shape_idx=*/{}, arg_idx));
  arguments.push_back(dst);
  TF_RETURN_IF_ERROR(CollectSliceInfo(
      buffer_assignment, fusion_instr,
      /*slice_instrs=*/absl::Span<HloInstruction*>(slice_instrs),
      /*offsets=*/offset_buffer_indices, orig_shapes, sliced_shapes,
      offset_byte_sizes, arg_idx, can_compute_indvar_on_host, fake_modules,
      call_graph));

  // Sanity checks.
  //  1. Expect atleast one slicing operation.
  //  2. Expect atleast one dynamic index operation iff the fusion is a
  //  dynamic-address-fusion.
  if (absl::c_all_of(slice_instrs, [&](HloInstruction* slice_instr) {
        return slice_instr == nullptr;
      })) {
    return absl::InternalError("Expected atleast one slicing operation");
  }
  bool isDynamic =
      absl::c_any_of(slice_instrs, [&](const HloInstruction* slice_instr) {
        return DynCastOrNull<HloDynamicIndexInstruction>(slice_instr) !=
               nullptr;
      });
  TF_ASSIGN_OR_RETURN(
      auto backend_config,
      fusion_instr.backend_config<xla::gpu::GpuBackendConfig>());
  const std::string fusion_name =
      backend_config.fusion_backend_config().custom_fusion_config().name();
  TF_RET_CHECK(isDynamic == (fusion_name == "dynamic_address_computation"))
      << "Dynamic index operation found in a fusion instruction that is not "
         "labelled dynamic_address_computation";

  int64_t replica_count = instr->GetModule()->config().replica_count();
  int64_t partition_count = instr->GetModule()->config().num_partitions();
  absl::Status implementable_status =
      NcclThunkType::CheckImplementable(instr, replica_count, partition_count);
  bool is_degenerate = GetNcclCollectiveConfig(instr, use_global_device_ids)
                           .IsDegenerate(replica_count, partition_count);
  Thunk::ThunkInfo thunk_info = Thunk::ThunkInfo::WithProfileAnnotation(instr);

  FusionEmissionResult result;
  std::vector<std::unique_ptr<BufferAllocation>> fake_allocations(2);
  if (isDynamic) {
    // Provide fake allocations for inputs and outputs. The dynamic-slice thunk
    // will own these allocations.
    unsigned fake_arg_idx = 0;
    int64_t operand_byte_size =
        ShapeUtil::ByteSizeOf(instr->operand(fake_arg_idx)->shape());
    fake_allocations[fake_arg_idx] = std::make_unique<BufferAllocation>(
        /*index=*/fake_arg_idx, operand_byte_size, /*color=*/0);
    src = BufferAllocation::Slice(
        /*allocation=*/fake_allocations[fake_arg_idx].get(), /*offset=*/0,
        /*size=*/operand_byte_size);
    fake_arg_idx++;
    TF_RET_CHECK(instr->shape().IsArray() &&
                 "The output is not expected to be a tuple.");
    int64_t out_fake_byte_size =
        ShapeUtil::ByteSizeOf(instr->shape());  // TODO: we don't need this
    fake_allocations[fake_arg_idx] = std::make_unique<BufferAllocation>(
        /*index=*/fake_arg_idx, /*size*/ out_fake_byte_size, /*color=*/0);
    dst = BufferAllocation::Slice(
        /*allocation=*/fake_allocations[fake_arg_idx].get(),
        /*offset=*/0, /*size=*/out_fake_byte_size);
  }

  // First we get the thunk sequence. This decides whether to generate a d2d
  // copy thunk or collective thunk.
  ThunkSequence seq;
  if (is_degenerate) {
    // Degenerate collectives are simply identity function. Buffer
    // assignment expects a copy, so that's what we do.
    const Shape shape = instr->operand(0)->shape();
    TF_RET_CHECK(shape == instr->shape())
        << "Expected operand shape to be equal to result shape, because "
           "the "
           "collective is degenerate: "
        << shape.ToString() << " vs " << instr->shape().ToString();
    seq.emplace_back(std::make_unique<DeviceToDeviceCopyThunk>(
        thunk_info,
        /*source_buffer=*/src,
        /*destination_buffer=*/dst,
        /*mem_size=*/ShapeUtil::ByteSizeOf(shape)));
  } else if (implementable_status.ok()) {
    std::vector<NcclCollectiveThunk::Buffer> buffers;
    const Shape& src_shape = instr->operand(0)->shape();
    const Shape& dst_shape = instr->shape();
    buffers.push_back(NcclCollectiveThunk::Buffer{
        /*element_count=*/ShapeUtil::ElementsIn(src_shape),
        /*source_buffer=*/src,
        /*destination_buffer=*/dst,
        /*source_memory_space=*/src_shape.layout().memory_space(),
        /*destination_memory_space=*/dst_shape.layout().memory_space(),
        /*source_value=*/nullptr,
        /*destination_value=*/nullptr});
    auto collective_start_thunk = std::make_unique<NcclThunkType>(
        thunk_info, NcclApi::Default(), instr, buffers);
    auto collective_done_thunk = std::make_unique<NcclCollectiveDoneThunk>(
        /*kind=*/collective_done_thunk_kind,
        /*thunk_info=*/Thunk::ThunkInfo::WithProfileAnnotation(instr),
        /*async_events=*/collective_start_thunk->async_events(),
        /*async_stream_kind=*/AsyncStreamKind::kCollective);
    seq.emplace_back(std::move(collective_start_thunk));
    seq.emplace_back(std::move(collective_done_thunk));
  } else {
    return implementable_status;
  }

  // Depending on whether this is a dynamic fusion or not, we wrap the thunk(s)
  // within a dynamic-slice thunk.
  if (isDynamic) {
    std::unique_ptr<Thunk> thunk = std::make_unique<DynamicSliceThunk>(
        thunk_info, std::make_unique<ThunkSequence>(std::move(seq)),
        std::move(arguments), std::move(fake_allocations),
        std::move(offset_buffer_indices), std::move(orig_shapes),
        std::move(sliced_shapes), std::move(offset_byte_sizes),
        std::move(fake_modules), std::move(init_module),
        std::move(update_module));
    result.thunks.push_back(std::move(thunk));
  } else {
    for (auto& thunk : seq) {
      result.thunks.push_back(std::move(thunk));
    }
  }
  return result;
}

}  // namespace

absl::StatusOr<FusionEmissionResult> CustomFusion::Emit(
    IrEmitterContext& ir_emitter_context,
    const HloFusionInstruction& fusion) const {
  TF_ASSIGN_OR_RETURN(auto gpu_config,
                      fusion.backend_config<GpuBackendConfig>());
  const FusionBackendConfig& backend_config =
      gpu_config.fusion_backend_config();
  const CustomFusionConfig& config = backend_config.custom_fusion_config();

  VLOG(3) << "Lower HLO fusion to a custom fusion " << config.name();

  auto* registry = CustomKernelFusionRegistry::Default();
  auto* custom_kernel_fusion = registry->Lookup(config.name());

  // If custom fusion is not found it means that some of the build targets might
  // not be statically linked into the binary.
  if (custom_kernel_fusion == nullptr) {
    return absl::InternalError(
        absl::StrCat("Custom kernel fusion ", config.name(),
                     " not found in a default registry."));
  }

  // Load custom kernels that can implement a fusion computation.
  TF_ASSIGN_OR_RETURN(std::vector<CustomKernel> kernels,
                      custom_kernel_fusion->LoadKernels(
                          ir_emitter_context.gpu_device_info(),
                          fusion.fused_instructions_computation()));

  // This should never happen, it means that compilation pipeline created a
  // fusion operation that is not supported by a given custom fusion.
  if (kernels.empty()) {
    return absl::InternalError(
        absl::StrCat("Custom kernel fusion ", config.name(),
                     " returned empty custom kernels for a fused computation"));
  }

  TF_ASSIGN_OR_RETURN(auto thunk,
                      BuildCustomKernelThunkForFusion(
                          ir_emitter_context, fusion,
                          std::move(kernels[config.kernel_index()])));

  FusionEmissionResult result;
  result.thunks.push_back(std::move(thunk));
  return result;
}

absl::StatusOr<FusionEmissionResult> DynamicSliceFusion::Emit(
    IrEmitterContext& ir_emitter_context,
    const HloFusionInstruction& fusion) const {
  const HloFusionAdaptor& adaptor = analysis_.fusion();
  // Only reduce-scatter is supported for now.
  auto maybe_collective =
      HloBfsFindIf(/*roots=*/adaptor.GetRoots(), /*fusion=*/adaptor,
                   /*visit=*/[](HloInstructionAdaptor node) -> bool {
                     return node.opcode() == HloOpcode::kReduceScatter;
                   });
  if (maybe_collective != std::nullopt) {
    const HloReduceScatterInstruction* rs =
        Cast<const HloReduceScatterInstruction>(
            &maybe_collective->instruction());
    return EmitCollective<NcclReduceScatterStartThunk,
                          HloReduceScatterInstruction>(
        ir_emitter_context, adaptor, /*fusion_instr=*/fusion, /*instr=*/rs,
        /*use_global_device_ids=*/rs->use_global_device_ids(), *call_graph_);
  }
  auto maybe_custom_call_adaptor = HloBfsFindIf(
      adaptor.GetRoots(), adaptor,
      [](auto node) { return node.opcode() == HloOpcode::kCustomCall; });
  if (maybe_custom_call_adaptor == std::nullopt) {
    return absl::InternalError("DynamicSliceFusion requires a CustomCall hero");
  }

  const auto& custom_call = *static_cast<const HloCustomCallInstruction*>(
      &maybe_custom_call_adaptor->instruction());
  if (IsLegacyCublasMatmul(custom_call)) {
    return EmitGemm(ir_emitter_context, adaptor, fusion, custom_call,
                    *call_graph_);
  }

  return EmitCustomCall(ir_emitter_context, adaptor, fusion, custom_call,
                        *call_graph_);
}

}  // namespace gpu
}  // namespace xla
