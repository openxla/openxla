/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_STREAM_EXECUTOR_TPU_TPU_OP_EXECUTABLE_H_
#define XLA_STREAM_EXECUTOR_TPU_TPU_OP_EXECUTABLE_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/tpu/c_api_decl.h"
#include "xla/stream_executor/tpu/tpu_executable_interface.h"
#include "xla/stream_executor/tpu/tpu_ops_c_api.h"
#include "tsl/framework/cancellation.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {

// An executable capable of being fed to a TPU device via TpuExecutor.
class TpuOpExecutable : public xla::TpuExecutableInterface {
 public:
  // Constructs an executable that holds a non-owning reference to an
  // XLA_TpuProgram.
  explicit TpuOpExecutable(
      const XLA_TpuProgram* core_program,
      std::unique_ptr<xla::HloModule> hlo_module,
      SE_OutsideCompilationParams* outside_compilation_params);

  ~TpuOpExecutable() override = default;

  const XLA_TpuProgram* core_program() const { return core_program_; }

  absl::string_view fingerprint() const override;

  tsl::StatusOr<xla::ExecutionOutput> ExecuteAsyncOnStream(
      const xla::ServiceExecutableRunOptions* run_options,
      std::vector<xla::ExecutionInput> arguments,
      xla::HloExecutionProfile* hlo_execution_profile) override;

 private:
  tsl::Status LoadProgramAndEnqueueToStream(
      const xla::ServiceExecutableRunOptions& run_options,
      absl::Span<const stream_executor::DeviceMemoryBase> arguments,
      stream_executor::DeviceMemoryBase result,
      const std::vector<stream_executor::DeviceMemoryBase>&
          cross_program_prefetch_addrs,
      const std::vector<uint32_t>& cross_program_prefetch_offsets) override;

  tsl::StatusOr<tsl::CancellationToken> RegisterCancellation(
      int device_ordinal);

  tsl::Status UnregisterCancellation(const tsl::Status& status,
                                     tsl::CancellationToken cancel_token,
                                     int device_ordinal, SE_Stream* c_stream);

  const XLA_TpuProgram* const core_program_;

  SE_OutsideCompilationParams* outside_compilation_params_;

  TF_DISALLOW_COPY_AND_ASSIGN(TpuOpExecutable);
};

}  // namespace tensorflow

#endif  // XLA_STREAM_EXECUTOR_TPU_TPU_OP_EXECUTABLE_H_
