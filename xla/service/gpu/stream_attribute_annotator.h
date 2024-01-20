/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_STREAM_ATTRIBUTE_ANNOTATOR_H_
#define XLA_SERVICE_GPU_STREAM_ATTRIBUTE_ANNOTATOR_H_

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/statusor.h"

namespace xla {
namespace gpu {

// This pass checks to see if there's any instruction, that
// consumes data from other computes streams, is missing
// wait_on_operation_queues attribute. It will annotate
// the corresponding instruction with the correct attribute
// in GpuBackendConfig.

class StreamAttributeAnnotator : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "stream-attribute-annotator";
  }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_STREAM_ATTRIBUTE_ANNOTATOR_H_
