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

#include "xla/service/gpu/model/gpu_indexing_performance_model.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/fusions/triton.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/coalescing_analysis.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/service/gpu/model/symbolic_tile_analysis.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/service/instruction_fusion.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

int64_t GpuPerformanceModelWithIndexingAnalysis::FlopsPerElement(
    const HloInstruction* instr) {
  // Instruction that are only used for indexing are not counted for FLOPs.
  switch (instr->opcode()) {
    case HloOpcode::kBitcast:
    case HloOpcode::kBroadcast:
    case HloOpcode::kConstant:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kGather:
    case HloOpcode::kIota:
    case HloOpcode::kPad:
    case HloOpcode::kParameter:
    case HloOpcode::kSlice:
    case HloOpcode::kTranspose:
    case HloOpcode::kTuple:
      return 0;
    default:
      break;
  };

  // Get the FLOPs per element for elementwise operations that only depend on
  // the element type.
  if (instr->IsElementwise()) {
    return cost_analysis_.GetFlopsPerElementwiseOpElement(
        instr->shape().element_type(), instr->opcode());
  }

  if (instr->opcode() == HloOpcode::kReduce) {
    int64_t flops_per_reduce_computation = 0;
    for (const HloInstruction* reducer_instr :
         instr->called_computations()[0]->instructions()) {
      flops_per_reduce_computation += FlopsPerElement(reducer_instr);
    }

    auto operand_shape = instr->operand(0)->shape();
    auto output_shape = instr->shape().IsArray()
                            ? instr->shape()
                            : instr->shape().tuple_shapes(0);

    // Size of reduction dimensions.
    int64_t reduction_factor = ShapeUtil::ElementsIn(operand_shape) /
                               ShapeUtil::ElementsIn(output_shape);

    // The Cost Model assumes that the reduction computation is applied N-1
    // times to reduce N elements. This is not true, because emitters will
    // generate a loop with N iterations. We don't fix it here to keep this
    // estimate consistent with GpuHloCostAnalysis. This is like doesn't matter
    // much for the application of the Cost Model.
    return (reduction_factor - 1) * flops_per_reduce_computation;
  }

  // Encountered unexpected instruction, call to GpuHloCostAnalysis.
  TF_CHECK_OK(
      cost_analysis_.RevisitInstruction(const_cast<HloInstruction*>(instr)));

  return cost_analysis_.flop_count(*instr) /
         ShapeUtil::ElementsInRecursive(instr->shape());
}

int64_t GpuPerformanceModelWithIndexingAnalysis::GetShapeSizeRecursive(
    const Shape& shape) const {
  CHECK(shape.IsArray() || shape.IsTuple());
  if (shape.IsArray()) {
    return shape_size_(shape);
  }

  int64_t total_size = 0;
  for (const auto& element_shape : shape.tuple_shapes()) {
    total_size += GetShapeSizeRecursive(element_shape);
  }
  return total_size;
}

int64_t GetIterationSpaceSize(const IndexingMap& indexing_map,
                              const HloInstruction* instr) {
  if (indexing_map.IsUndefined()) {
    return ShapeUtil::ElementsInRecursive(instr->shape());
  }

  if (indexing_map.IsKnownEmpty()) {
    return 0;
  }

  auto get_ranges_iteration_space_size =
      [](const std::vector<Interval>& ranges) {
        int64_t num_iters = 1;
        for (const Interval& range : ranges) {
          num_iters *= range.upper - range.lower + 1;
        }
        return num_iters;
      };

  return get_ranges_iteration_space_size(indexing_map.GetSymbolBounds()) *
         get_ranges_iteration_space_size(indexing_map.GetDimensionBounds());
}

EstimateRunTimeData
GpuPerformanceModelWithIndexingAnalysis::EstimateRunTimeForFusion(
    const HloFusionAnalysis& fusion_analysis, bool is_coalesced) {
  auto& fusion_adaptor = fusion_analysis.fusion();
  VLOG(5) << "EstimateRunTimeForFusion: " << fusion_adaptor.ToString();

  auto roots = fusion_adaptor.GetRoots();
  CHECK_EQ(roots.size(), 1)
      << "Indexing cost model doesn't support multi-output fusions.";
  auto root_shape = roots.front().shape();

  LaunchDimensions launch_dimensions =
      EstimateFusionLaunchDimensions(fusion_analysis);

  int64_t num_blocks = launch_dimensions.num_blocks();

  // Compute indexing from root to each instruction in the fusion and fusion
  // operands. For each instruction, tells which elements of the instructions
  // result will be used to compute one result element of the fusion.
  auto grouped_fusion_indexing = ComputeGroupedOutputToInputIndexing(
      fusion_adaptor, roots[0], mlir_context_);

  int64_t flops = 0;
  int64_t bytes_read = 0;
  absl::Duration read_time = absl::ZeroDuration();

  for (const auto& [instr, indexing_maps] : grouped_fusion_indexing) {
    VLOG(10) << "instr: " << instr->name();

    // Instructions inside the fusion are computation and account for FLOPs
    // count. Instructions outside the fusion are operands of the fusion and
    // account for memory read time.
    bool is_operand = !fusion_adaptor.ContainsInstruction(instr);

    auto element_type = instr->shape().element_type();
    int64_t n_bytes_total = 0;
    for (const auto& indexing_map : indexing_maps) {
      VLOG(10) << indexing_map.ToString();

      int64_t num_iters = GetIterationSpaceSize(indexing_map, instr);

      if (is_operand) {
        int64_t type_size = ShapeUtil::ByteSizeOfPrimitiveType(element_type);
        n_bytes_total += type_size * num_iters;
      } else {
        int64_t flops_per_element = FlopsPerElement(instr);
        flops += flops_per_element * num_iters;
      }
    }

    if (is_operand) {
      int64_t operand_size = shape_size_(instr->shape());
      int64_t n_bytes_net = std::min(operand_size, n_bytes_total);
      bytes_read += n_bytes_total;

      VLogOperandRead(instr, n_bytes_total, n_bytes_net, is_coalesced);

      read_time +=
          ReadTimeWithDRAMHeuristic(*device_info_, num_blocks, n_bytes_net,
                                    n_bytes_total, element_type, is_coalesced);
    }
  }

  int64_t bytes_written = GetShapeSizeRecursive(root_shape);

  absl::Duration compute_time =
      ComputeTime(*device_info_, flops, num_blocks,
                  launch_dimensions.num_threads_per_block());
  absl::Duration write_time = WriteTime(*device_info_, bytes_written);
  absl::Duration memory_access_time = read_time + write_time;
  absl::Duration exec_time = CombineComputeAndMemoryAccessTime(
      compute_time, memory_access_time,
      GpuPerformanceModelOptions::PriorityFusion());

  EstimateRunTimeData runtime_data = {flops,     bytes_read, bytes_written,
                                      read_time, write_time, compute_time,
                                      exec_time};
  VLOG(3) << "Runtime data for HLO fusion: " << fusion_adaptor.ToString()
          << "\n"
          << launch_dimensions.ToString() << "\n"
          << runtime_data.ToString();

  return runtime_data;
}

EstimateRunTimeData
GpuPerformanceModelWithIndexingAnalysis::EstimateRunTimeForInstruction(
    const HloInstruction* producer) {
  // Stand-alone bitcast is always no-op during runtime.
  if (producer->opcode() == HloOpcode::kBitcast) {
    return EstimateRunTimeData{/*flops=*/0,
                               /*bytes_read=*/0,
                               /*bytes_written=*/0,
                               /*read_time=*/absl::ZeroDuration(),
                               /*write_time=*/absl::ZeroDuration(),
                               /*compute_time=*/absl::ZeroDuration(),
                               /*exec_time=*/absl::ZeroDuration()};
  }

  auto fusion_analysis = AnalyzeFusion(*producer, *device_info_);

  bool is_coalesced = IsReadCoalescedHeuristic(
      fusion_analysis.GetEmitterFusionKind(), producer);
  return EstimateRunTimeForFusion(fusion_analysis, is_coalesced);
}

EstimateRunTimeData
GpuPerformanceModelWithIndexingAnalysis::EstimateRunTimeForProducerConsumer(
    const HloInstruction* producer, const HloInstruction* consumer) {
  auto fusion_analysis =
      AnalyzeProducerConsumerFusion(*producer, *consumer, *device_info_);

  bool is_coalesced = IsReadCoalescedHeuristic(
      fusion_analysis.GetEmitterFusionKind(), producer, consumer);
  return EstimateRunTimeForFusion(fusion_analysis, is_coalesced);
}

/*static*/
GpuPerformanceModelWithIndexingAnalysis::RunTimes
GpuPerformanceModelWithIndexingAnalysis::EstimateRunTimes(
    const HloInstruction* producer,
    absl::Span<const HloInstruction* const> fused_consumers) {
  auto producer_runtime = EstimateRunTimeForInstruction(producer);

  absl::Duration time_unfused =
      kKernelLaunchOverhead * (fused_consumers.size() + 1) +
      producer_runtime.exec_time;

  absl::Duration time_fused = kKernelLaunchOverhead * fused_consumers.size();

  for (const auto& consumer : fused_consumers) {
    time_unfused += EstimateRunTimeForInstruction(consumer).exec_time;
    time_fused +=
        EstimateRunTimeForProducerConsumer(producer, consumer).exec_time;
  }

  return {time_unfused, time_fused};
}

absl::StatusOr<EstimateRunTimeData>
GpuPerformanceModelWithIndexingAnalysis::EstimateRunTimeForTiledHloComputation(
    const HloFusionAdaptor& fusion_adaptor,
    const TiledHloComputation& tiled_hlo_computation,
    const LaunchDimensions& launch_dimensions) {
  absl::flat_hash_map<const HloInstruction*, int64_t> n_bytes_total_map;

  int64_t flops = 0;
  int64_t bytes_read = 0;
  int64_t num_blocks = launch_dimensions.num_blocks();

  for (const auto& tiled_hlo : tiled_hlo_computation.instructions()) {
    // Total number of elements that are read from memory or computed for this
    // tile across all blocks.
    int64_t num_elements = num_blocks * Product(tiled_hlo->tile_sizes());

    const HloInstruction* hlo = tiled_hlo->hlo();

    if (hlo->opcode() == HloOpcode::kConcatenate) {
      // TODO(b/351342921): Add propagation of the number of blocks that read or
      // compute a tile. Concatenate is the only operation that may change that.
      return absl::FailedPreconditionError(
          "Concatenate is not supported by the indexing cost model.");
    }

    if (fusion_adaptor.ContainsInstruction(hlo)) {
      // Tiles inside the computation contribute to the total FLOPs count.
      flops += FlopsPerElement(hlo) * num_elements;
    } else {
      // Tiles of the operands of the fusion contribute to the total memory
      // read time.
      int64_t element_type_size =
          ShapeUtil::ByteSizeOfPrimitiveType(hlo->shape().element_type());
      int64_t tile_bytes_read = element_type_size * num_elements;

      bytes_read += tile_bytes_read;
      n_bytes_total_map[hlo] += tile_bytes_read;
    }
  }

  absl::Duration read_time = absl::ZeroDuration();
  for (const auto& [hlo, n_bytes_total] : n_bytes_total_map) {
    int64_t operand_size = shape_size_(hlo->shape());
    int64_t n_bytes_net = std::min(operand_size, n_bytes_total);

    read_time += ReadTimeWithDRAMHeuristic(
        *device_info_, num_blocks, n_bytes_net, n_bytes_total,
        /*element_type=*/hlo->shape().element_type(),
        /*coalesced=*/true);
  }

  int64_t bytes_written =
      GetShapeSizeRecursive(tiled_hlo_computation.GetRoot()->hlo()->shape());

  absl::Duration compute_time =
      ComputeTime(*device_info_, flops, launch_dimensions.num_blocks(),
                  launch_dimensions.num_threads_per_block());
  absl::Duration write_time = WriteTime(*device_info_, bytes_written);
  absl::Duration memory_access_time = read_time + write_time;
  absl::Duration exec_time = CombineComputeAndMemoryAccessTime(
      compute_time, memory_access_time,
      GpuPerformanceModelOptions::PriorityFusion());

  return EstimateRunTimeData{/*flops=*/flops,
                             /*bytes_read=*/bytes_read,
                             /*bytes_written=*/bytes_written,
                             /*read_time=*/read_time,
                             /*write_time=*/write_time,
                             /*compute_time=*/compute_time,
                             /*exec_time=*/exec_time};
}

absl::StatusOr<EstimateRunTimeData>
GpuPerformanceModelWithIndexingAnalysis::EstimateRunTimeForTiledFusion(
    const HloFusionAdaptor& fusion_adaptor,
    const LaunchDimensions& launch_dimensions,
    absl::Span<const int64_t> tile_sizes) {
  // TODO(b/332714755): Add caching for SymbolicTileAnalysis.
  SymbolicTileAnalysisOrError analysis_or_error =
      SymbolicTileAnalysis::AnalyzeFusion(fusion_adaptor, mlir_context_);
  if (const auto* fusion_decision =
          std::get_if<FusionDecision>(&analysis_or_error)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "SymbolicTileAnalysis failed. ", fusion_decision->Explain()));
  }
  SymbolicTileAnalysis analysis =
      std::get<SymbolicTileAnalysis>(std::move(analysis_or_error));

  TF_ASSIGN_OR_RETURN(TiledHloComputation tiled_hlo_computation,
                      analysis.ComputeTiledHloInstructions(tile_sizes));

  return EstimateRunTimeForTiledHloComputation(
      fusion_adaptor, tiled_hlo_computation, launch_dimensions);
}

absl::StatusOr<EstimateRunTimeData>
GpuPerformanceModelWithIndexingAnalysis::EstimateRunTimeForTriton(
    const HloInstruction* producer, const HloInstruction* consumer) {
  const auto& fusion_analysis =
      (consumer == nullptr) ? fusion_analysis_cache_->Get(*producer)
                            : fusion_analysis_cache_->Get(*producer, *consumer);
  auto launch_config = TritonFusion(fusion_analysis).launch_config();

  if (!launch_config.has_value()) {
    return absl::InvalidArgumentError(
        "Could not get launch config for Triton fusion.");
  }

  return EstimateRunTimeForTiledFusion(
      fusion_analysis.fusion(), launch_config->launch_dimensions,
      launch_config->block_level_parameters.output_tile_sizes);
}

// Returns the number of warps to use based on the tile size. The numbers were
// originally selected from Triton SoftMax reduction row length.
// TODO(b/332714755): Make it smarter.
int64_t GetNumWarps(int64_t tile_size) {
  if (tile_size <= 512) return 1;
  if (tile_size <= 1024) return 2;
  if (tile_size <= 16384) return 4;
  if (tile_size <= 32768) return 8;
  if (tile_size <= 65536) return 16;
  return 32;
}

LaunchDimensions GetLaunchDimensionsForTiledFusion(
    const TiledHloComputation& tiled_hlo_computation) {
  const auto* tiled_root = tiled_hlo_computation.GetRoot();
  int64_t num_blocks = tiled_hlo_computation.num_output_tiles();
  int64_t num_warps = GetNumWarps(Product(tiled_root->tile_sizes()));

  return {static_cast<uint64_t>(num_blocks),
          static_cast<uint64_t>(num_warps * WarpSize())};
}

absl::StatusOr<TiledRunTimeDataOrError>
GpuPerformanceModelWithIndexingAnalysis::TryFindBestTilingForFusion(
    const HloFusionAdaptor& fusion_adaptor) {
  SymbolicTileAnalysisOrError analysis_or_error =
      SymbolicTileAnalysis::AnalyzeFusion(fusion_adaptor, mlir_context_);

  if (const auto* fusion_decision =
          std::get_if<FusionDecision>(&analysis_or_error)) {
    return *fusion_decision;
  }

  SymbolicTileAnalysis analysis =
      std::get<SymbolicTileAnalysis>(std::move(analysis_or_error));

  TF_ASSIGN_OR_RETURN(auto tilings, analysis.GetGoodTilings());

  std::optional<TiledRunTimeData> best_tiled_run_time_data;

  for (const auto& tiling : tilings) {
    TF_ASSIGN_OR_RETURN(TiledHloComputation tiled_hlo_computation,
                        analysis.ComputeTiledHloInstructions(tiling));

    LaunchDimensions launch_dimensions =
        GetLaunchDimensionsForTiledFusion(tiled_hlo_computation);

    TF_ASSIGN_OR_RETURN(
        EstimateRunTimeData estimate_run_time_data,
        EstimateRunTimeForTiledHloComputation(
            fusion_adaptor, tiled_hlo_computation, launch_dimensions));

    if (!best_tiled_run_time_data.has_value() ||
        estimate_run_time_data.exec_time <
            best_tiled_run_time_data->runtime_data.exec_time) {
      BlockLevelParameters block_level_parameters;
      block_level_parameters.output_tile_sizes =
          std::vector<int64_t>(tiling.begin(), tiling.end());
      block_level_parameters.num_warps =
          launch_dimensions.num_threads_per_block() / WarpSize();

      best_tiled_run_time_data =
          TiledRunTimeData{estimate_run_time_data, block_level_parameters};
    }
  }

  if (!best_tiled_run_time_data.has_value()) {
    return FusionDecision("No valid tilings found.");
  }
  return *best_tiled_run_time_data;
}

}  // namespace gpu
}  // namespace xla
