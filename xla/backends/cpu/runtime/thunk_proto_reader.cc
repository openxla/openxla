/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/cpu/runtime/thunk_proto_reader.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/backends/cpu/runtime/all_gather_thunk.h"
#include "xla/backends/cpu/runtime/all_reduce_thunk.h"
#include "xla/backends/cpu/runtime/all_to_all_thunk.h"
#include "xla/backends/cpu/runtime/call_thunk.h"
#include "xla/backends/cpu/runtime/collective_permute_thunk.h"
#include "xla/backends/cpu/runtime/conditional_thunk.h"
#include "xla/backends/cpu/runtime/convolution_thunk.h"
#include "xla/backends/cpu/runtime/copy_thunk.h"
#include "xla/backends/cpu/runtime/custom_call_thunk.h"
#include "xla/backends/cpu/runtime/dot_thunk.h"
#include "xla/backends/cpu/runtime/fft_thunk.h"
#include "xla/backends/cpu/runtime/infeed_thunk.h"
#include "xla/backends/cpu/runtime/kernel_thunk.h"
#include "xla/backends/cpu/runtime/outfeed_thunk.h"
#include "xla/backends/cpu/runtime/reduce_scatter_thunk.h"
#include "xla/backends/cpu/runtime/rng_state_thunk.h"
#include "xla/backends/cpu/runtime/sort_thunk.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk.pb.h"
#include "xla/backends/cpu/runtime/topk_thunk.h"
#include "xla/backends/cpu/runtime/while_thunk.h"
#include "xla/backends/cpu/runtime/xnnpack/xnn_dot_thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::cpu::thunk_reader {

absl::StatusOr<std::unique_ptr<Thunk>> ThunkFromProto(
    const ThunkProto& proto, const BufferAssignment& buffer_assignment) {
  Thunk::Kind kind = Thunk::ProtoThunkToThunkKind(proto);
  if (Thunk::KindToString(kind) != proto.kind()) {
    return absl::Status(
        absl::StatusCode::kInvalidArgument,
        absl::StrFormat(
            "Kind mismatch between proto kind `%s` and thunk kind `%s`.",
            proto.kind(), Thunk::KindToString(kind)));
  }

  switch (kind) {
    case Thunk::Kind::kAllGather:
      return AllGatherThunk::FromProto(proto, buffer_assignment);
    case Thunk::Kind::kAllReduce:
      return AllReduceThunk::FromProto(proto, buffer_assignment);
    case Thunk::Kind::kAllToAll:
      return AllToAllThunk::FromProto(proto, buffer_assignment);
    case Thunk::Kind::kCollectivePermute:
      return CollectivePermuteThunk::FromProto(proto, buffer_assignment);
    case Thunk::Kind::kReduceScatter:
      return ReduceScatterThunk::FromProto(proto, buffer_assignment);
    case Thunk::Kind::kCall: {
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<ThunkSequence> call_sequence,
          ThunkSequenceFromProto(proto.call_thunk().called_sequence(),
                                 buffer_assignment));
      return CallThunk::FromProto(proto, buffer_assignment,
                                  std::move(*call_sequence));
    }
    case Thunk::Kind::kConditional: {
      std::vector<ThunkSequence> branch_sequences;
      for (const ThunkSequenceProto& branch_sequence_proto :
           proto.conditional_thunk().branch_sequences()) {
        TF_ASSIGN_OR_RETURN(
            std::unique_ptr<ThunkSequence> branch_sequence,
            ThunkSequenceFromProto(branch_sequence_proto, buffer_assignment));
        branch_sequences.push_back(std::move(*branch_sequence));
      }
      return ConditionalThunk::FromProto(proto, buffer_assignment,
                                         std::move(branch_sequences));
    }
    case Thunk::Kind::kConvolution:
      return ConvolutionThunk::FromProto(proto, buffer_assignment);
    case Thunk::Kind::kCopy:
      return CopyThunk::FromProto(proto, buffer_assignment);
    case Thunk::Kind::kCustomCall:
      return CustomCallThunk::FromProto(proto, buffer_assignment);
    case Thunk::Kind::kDot:
      return DotThunk::FromProto(proto, buffer_assignment);
    case Thunk::Kind::kFft:
      return FftThunk::FromProto(proto, buffer_assignment);
    case Thunk::Kind::kInfeed:
      return InfeedThunk::FromProto(proto, buffer_assignment);
    case Thunk::Kind::kKernel:
      return KernelThunk::FromProto(proto, buffer_assignment);
    case Thunk::Kind::kOutfeed:
      return OutfeedThunk::FromProto(proto, buffer_assignment);
    case Thunk::Kind::kRngGetAndUpdateState:
      return RngGetAndUpdateStateThunk::FromProto(proto, buffer_assignment);
    case Thunk::Kind::kSort:
      return SortThunk::FromProto(proto, buffer_assignment);
    case Thunk::Kind::kTopK:
      return TopKThunk::FromProto(proto, buffer_assignment);
    case Thunk::Kind::kWhile: {
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<ThunkSequence> cond_sequence,
          ThunkSequenceFromProto(proto.while_thunk().cond_sequence(),
                                 buffer_assignment));
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<ThunkSequence> body_sequence,
          ThunkSequenceFromProto(proto.while_thunk().body_sequence(),
                                 buffer_assignment));
      return WhileThunk::FromProto(proto, buffer_assignment,
                                   std::move(*cond_sequence),
                                   std::move(*body_sequence));
    }
    case Thunk::Kind::kXnnFusion:
      // TODO(basioli) follow up CL should add XnnDotThunk, we should
      // abide by the same inheritance pattern as the other thunks.
      return XnnDotThunk::FromProto(proto, buffer_assignment);
    default:
      return absl::Status(absl::StatusCode::kInvalidArgument,
                          absl::StrFormat("Unsupported thunk kind: %s",
                                          Thunk::KindToString(kind)));
  }
}

absl::StatusOr<std::unique_ptr<ThunkSequence>> ThunkSequenceFromProto(
    const ThunkSequenceProto& proto,
    const BufferAssignment& buffer_assignment) {
  auto thunk_sequence = std::make_unique<ThunkSequence>();
  for (const ThunkProto& thunk_proto : proto.thunks()) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<Thunk> thunk,
                        ThunkFromProto(thunk_proto, buffer_assignment));
    thunk_sequence->push_back(std::move(thunk));
  }
  return thunk_sequence;
}

}  // namespace xla::cpu::thunk_reader
