func.func @foo(%lhs: tensor<?x?xi8>, %rhs: tensor<?x?xi8>) -> tensor<?x?xi32> {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %m = tensor.dim %lhs, %c0 : tensor<?x?xi8>
  %n = tensor.dim %rhs, %c0 : tensor<?x?xi8>
  %empty = tensor.empty(%m, %n) : tensor<?x?xi32>
  %zero = linalg.fill ins(%c0_i32 : i32) outs(%empty : tensor<?x?xi32>) -> tensor<?x?xi32>
  %result = linalg.matmul ins(%lhs, %rhs: tensor<?x?xi8>, tensor<?x?xi8>) outs(%zero: tensor<?x?xi32>) -> tensor<?x?xi32>
  return %result: tensor<?x?xi32>
}