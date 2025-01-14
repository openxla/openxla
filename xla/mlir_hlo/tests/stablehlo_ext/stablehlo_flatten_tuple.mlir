// RUN: mlir-hlo-opt -split-input-file -stablehlo-ext-flatten-tuple %s | FileCheck %s

// TODO(agunjal): Original test expects canonicalization to happen after the pass run.
// CHECK-LABEL: @custom_call
// CHECK-SAME: %[[X:.*]]: tensor<6x3xf32>
// CHECK: %[[CALL:.+]]:2 = stablehlo.custom_call @f(%[[X]]) {api_version = 2 : i32} : (tensor<6x3xf32>) -> (tensor<6xf32>, tensor<3xf32>) 
// COM: // CHECK: return %[[CALL]]#0, %[[CALL]]#1 : tensor<6xf32>, tensor<3xf32> 
func.func @custom_call(%x: tensor<6x3xf32>) -> (tensor<6xf32>, tensor<3xf32>) {
  %0 = "stablehlo.custom_call"(%x) {api_version = 2 : i32, call_target_name = "f"}
    : (tensor<6x3xf32>) -> tuple<tensor<6xf32>, tensor<3xf32>>
  %1 = "stablehlo.get_tuple_element"(%0) {index = 0 : i32} : (tuple<tensor<6xf32>, tensor<3xf32>>) -> tensor<6xf32>
  %2 = "stablehlo.get_tuple_element"(%0) {index = 1 : i32} : (tuple<tensor<6xf32>, tensor<3xf32>>) -> tensor<3xf32>
  return %1, %2 : tensor<6xf32>, tensor<3xf32>
}

// -----

// CHECK-LABEL: @custom_call_tupled_operand
// COM: // CHECK-NOT: stablehlo.tuple
func.func @custom_call_tupled_operand(%arg: tuple<tensor<ui32>, tensor<i32>>)
  -> (tensor<i32>, tensor<ui32>) {
  %0 = stablehlo.constant dense<1> : tensor<ui32>
  %1 = stablehlo.constant dense<10> : tensor<i32>
  %2 = stablehlo.tuple %0, %1, %arg : tuple<tensor<ui32>, tensor<i32>,
                                       tuple<tensor<ui32>, tensor<i32>>>
  %3 = stablehlo.custom_call @ScalarProgramDummyConstant(%2)
    : (tuple<tensor<ui32>, tensor<i32>, tuple<tensor<ui32>, tensor<i32>>>)
    -> tensor<ui32>
  return %1, %3 : tensor<i32>, tensor<ui32>
}
