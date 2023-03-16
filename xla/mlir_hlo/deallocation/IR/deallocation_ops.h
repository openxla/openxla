/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_MLIR_HLO_DEALLOCATION_IR_DEALLOCATION_OPS_H_
#define XLA_MLIR_HLO_DEALLOCATION_IR_DEALLOCATION_OPS_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "deallocation/IR/deallocation_dialect.h.inc"
#include "deallocation/IR/deallocation_ops.h.inc"
#undef GET_OP_CLASSES

#endif  // XLA_MLIR_HLO_DEALLOCATION_IR_DEALLOCATION_OPS_H_
