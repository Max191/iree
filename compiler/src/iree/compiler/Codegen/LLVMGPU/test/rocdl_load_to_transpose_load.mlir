// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(func.func(iree-rocdl-load-to-transpose-load))' %s | FileCheck %s

// Test 1: Simple 2D transfer_read with hint (should drop hint for now)
// CHECK-LABEL: func.func @transfer_read_with_hint_2d
func.func @transfer_read_with_hint_2d(%src: memref<128x256xf16, #gpu.address_space<workgroup>>) -> vector<4xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %tid = gpu.thread_id x

  // Hint wrapping the indices
  %hint:2 = iree_gpu.transpose_load_index_hint %c0, %tid : index, index

  %cst = arith.constant 0.0 : f16
  // CHECK-NOT: iree_gpu.transpose_load_index_hint
  // CHECK: vector.transfer_read
  // CHECK-SAME: %{{.*}}[%c0, %{{.*}}]
  %0 = vector.transfer_read %src[%hint#0, %hint#1], %cst {in_bounds = [true]}
       : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<4xf16>
  return %0 : vector<4xf16>
}

// -----

// Test 2: Transfer_read with 3D hint (row indices + column)
// CHECK-LABEL: func.func @transfer_read_with_hint_3d
func.func @transfer_read_with_hint_3d(%src: memref<128x256x512xf16, #gpu.address_space<workgroup>>)
    -> vector<4xf16> {
  %c0 = arith.constant 0 : index
  %tid_x = gpu.thread_id x
  %tid_y = gpu.thread_id y

  // Hint with multiple row dimensions and column dimension
  %hint:3 = iree_gpu.transpose_load_index_hint %c0, %tid_y, %tid_x : index, index, index

  %cst = arith.constant 0.0 : f16
  // CHECK-NOT: iree_gpu.transpose_load_index_hint
  // CHECK: vector.transfer_read
  // CHECK-SAME: %{{.*}}[%c0, %{{.*}}, %{{.*}}]
  %0 = vector.transfer_read %src[%hint#0, %hint#1, %hint#2], %cst {in_bounds = [true]}
       : memref<128x256x512xf16, #gpu.address_space<workgroup>>, vector<4xf16>
  return %0 : vector<4xf16>
}

// -----

// Test 3: Transfer_read without hint (should be unchanged)
// CHECK-LABEL: func.func @transfer_read_no_hint
func.func @transfer_read_no_hint(%src: memref<128x256xf16, #gpu.address_space<workgroup>>) -> vector<4xf16> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f16
  // CHECK: vector.transfer_read
  %0 = vector.transfer_read %src[%c0, %c0], %cst {in_bounds = [true]}
       : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<4xf16>
  return %0 : vector<4xf16>
}

// -----

// Test 4: Orphaned hint op (hint not used, should be dropped)
// CHECK-LABEL: func.func @orphaned_hint
func.func @orphaned_hint() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // CHECK-NOT: iree_gpu.transpose_load_index_hint
  %hint:2 = iree_gpu.transpose_load_index_hint %c0, %c1 : index, index
  // Hint is not used for anything
  return
}

// -----

// Test 5: Multiple hints with different sizes
// CHECK-LABEL: func.func @multiple_hints
func.func @multiple_hints(%src: memref<128x256xf16, #gpu.address_space<workgroup>>) -> vector<4xf16> {
  %c0 = arith.constant 0 : index
  %tid_x = gpu.thread_id x
  %tid_y = gpu.thread_id y

  // First hint with 2 results
  %hint1:2 = iree_gpu.transpose_load_index_hint %c0, %tid_x : index, index

  // Second hint with 3 results
  %hint2:3 = iree_gpu.transpose_load_index_hint %hint1#0, %tid_y, %hint1#1 : index, index, index

  %cst = arith.constant 0.0 : f16
  // CHECK-NOT: iree_gpu.transpose_load_index_hint
  // CHECK: vector.transfer_read
  // CHECK-SAME: %{{.*}}[%c0, %{{.*}}]
  %0 = vector.transfer_read %src[%hint2#0, %hint2#2], %cst {in_bounds = [true]}
       : memref<128x256xf16, #gpu.address_space<workgroup>>, vector<4xf16>
  return %0 : vector<4xf16>
}