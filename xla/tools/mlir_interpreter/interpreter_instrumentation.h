/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_TOOLS_MLIR_INTERPRETER_INTERPRETER_INSTRUMENTATION_H_
#define TENSORFLOW_COMPILER_XLA_TOOLS_MLIR_INTERPRETER_INTERPRETER_INSTRUMENTATION_H_

#include "llvm/ADT/SmallVector.h"
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassInstrumentation.h"  // from @llvm-project
#include "xla/tools/mlir_interpreter/interpreter_value.h"

namespace mlir {
namespace interpreter {

// Instrumentation that runs the interpreter on random inputs after each pass.
// Reports changed results.
class MlirInterpreterInstrumentation : public PassInstrumentation {
 public:
  void runAfterPass(Pass* pass, Operation* op) override;

 private:
  llvm::SmallVector<InterpreterValue> reference_results_;
};

}  // namespace interpreter
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_XLA_TOOLS_MLIR_INTERPRETER_INTERPRETER_INSTRUMENTATION_H_
