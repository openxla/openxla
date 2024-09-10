/* Copyright 2019 The OpenXLA Authors.

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

// This file contains APIs that assume a StreamExecutor is backed by a GPU
// implementation. It ensures the underlying GPU context is active.

#ifndef XLA_STREAM_EXECUTOR_GPU_GPU_ACTIVATION_H_
#define XLA_STREAM_EXECUTOR_GPU_GPU_ACTIVATION_H_

#include "xla/stream_executor/gpu/scoped_activate_context.h"

namespace stream_executor {

class StreamExecutor;

namespace gpu {

class GpuExecutor;

// Activates a context within an enclosing scope.
class ScopedActivateExecutorContext {
 public:
  // Form that takes a GPU executor.
  explicit ScopedActivateExecutorContext(GpuExecutor* gpu_exec);

  // Form that takes an executor and extracts a GPU Executor --
  // fatal failure if it is not a GPU executor.
  explicit ScopedActivateExecutorContext(StreamExecutor* stream_exec);

  ~ScopedActivateExecutorContext();

 private:
  ScopedActivateContext* driver_scoped_activate_context_;

  ScopedActivateExecutorContext(const ScopedActivateExecutorContext&) = delete;
  void operator=(const ScopedActivateExecutorContext&) = delete;
};

}  // namespace gpu
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_GPU_GPU_ACTIVATION_H_
