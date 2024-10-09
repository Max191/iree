func.func @foo(%lhs: tensor<?x?xf16>, %rhs: tensor<?x?xf16>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c0_f32 = arith.constant 0.0 : f32
  %m = tensor.dim %lhs, %c0 : tensor<?x?xf16>
  %n = tensor.dim %rhs, %c0 : tensor<?x?xf16>
  %empty = tensor.empty(%m, %n) : tensor<?x?xf32>
  %zero = linalg.fill ins(%c0_f32 : f32) outs(%empty : tensor<?x?xf32>) -> tensor<?x?xf32>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<?x?xf16>, tensor<?x?xf16>) outs(%zero: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %result: tensor<?x?xf32>
}
