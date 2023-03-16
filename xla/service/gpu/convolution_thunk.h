/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_CONVOLUTION_THUNK_H_
#define XLA_SERVICE_GPU_CONVOLUTION_THUNK_H_

#include <optional>

#include "absl/container/flat_hash_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_conv_runner.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/thunk.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/status.h"

namespace xla {
namespace gpu {

// This class stores everything that StreamExecutor needs to launch a DNN
// convolution. It is generated by IrEmitter.
//
// This is thread-compatible.
class ConvolutionThunk : public Thunk {
 public:
  // Constructs a thunk for launching a DNN convolution.
  //
  // operand_slices should be in the same order as cudnn_call->operands().
  ConvolutionThunk(ThunkInfo thunk_info, GpuConvConfig config,
                   std::vector<BufferAllocation::Slice> operand_slices,
                   BufferAllocation::Slice result_slice,
                   BufferAllocation::Slice scratch_slice);

  ConvolutionThunk(const ConvolutionThunk&) = delete;
  ConvolutionThunk& operator=(const ConvolutionThunk&) = delete;

  Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  std::vector<BufferAllocation::Slice> operand_buffers_;
  BufferAllocation::Slice result_buffer_;
  BufferAllocation::Slice scratch_buffer_;
  MaybeFusedConvRunner& GetOrCreateRunner(
      const stream_executor::Stream* stream);

  // Convolution config
  const GpuConvConfig config_;
  absl::Mutex mu_;
  absl::flat_hash_map<const stream_executor::Stream*,
                      std::unique_ptr<MaybeFusedConvRunner>>
      runner_cache_ ABSL_GUARDED_BY(mu_);
};

// Launches the kernel that reorders input data for int8x32 convolutions.
class ConvolutionReorderThunk : public Thunk {
 public:
  ConvolutionReorderThunk(ThunkInfo thunk_info, absl::Span<int64_t> filter_nchw,
                          std::vector<BufferAllocation::Slice> operand_slices,
                          std::vector<BufferAllocation::Slice> result_slices);

  ConvolutionReorderThunk(const ConvolutionReorderThunk&) = delete;
  ConvolutionReorderThunk& operator=(const ConvolutionReorderThunk&) = delete;

  Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  static se::dnn::FilterDescriptor CreateFilterDescriptor(
      absl::Span<int64_t> filter_nchw);

  const se::dnn::FilterDescriptor filter_descriptor_;
  std::vector<BufferAllocation::Slice> operand_buffers_;
  std::vector<BufferAllocation::Slice> result_buffers_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_CONVOLUTION_THUNK_H_
