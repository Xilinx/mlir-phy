// RUN: phy-opt --convert-layout-to-physical="device=aie" %s | FileCheck %s
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// CHECK-LABEL: module @MM_2x2
module @MM_2x2 {

  %QA_0   = spatial.queue<1>(): !spatial.queue<memref<1024xi32>>
  %QA_1   = spatial.queue<1>(): !spatial.queue<memref<1024xi32>>
  %QB_0_0 = spatial.queue<1>(): !spatial.queue<memref<1024xi32>>
  %QB_0_1 = spatial.queue<1>(): !spatial.queue<memref<1024xi32>>
  %QB_1_0 = spatial.queue<1>(): !spatial.queue<memref<1024xi32>>
  %QB_1_1 = spatial.queue<1>(): !spatial.queue<memref<1024xi32>>
  %QC_0   = spatial.queue<1>(): !spatial.queue<memref<1024xi32>>
  %QC_1   = spatial.queue<1>(): !spatial.queue<memref<1024xi32>>

  %S_0_0  = spatial.queue<1>(): !spatial.queue<memref<1024xi32>>
  %S_0_1  = spatial.queue<1>(): !spatial.queue<memref<1024xi32>>
  %S_1_0  = spatial.queue<1>(): !spatial.queue<memref<1024xi32>>
  %S_1_1  = spatial.queue<1>(): !spatial.queue<memref<1024xi32>>

  func.func private @extern_kernel(%A: memref<1024xi32>, %B: memref<1024xi32>, %acc: memref<1024xi32>, %C: memref<1024xi32>) -> ()
  func.func private @kernel(%A: !spatial.queue<memref<1024xi32>>, %B: !spatial.queue<memref<1024xi32>>, %S: !spatial.queue<memref<1024xi32>>, %C: !spatial.queue<memref<1024xi32>>) {
    cf.br ^bb
^bb:
    %aA = spatial.front(%A): memref<1024xi32>
    %aB = spatial.front(%B): memref<1024xi32>
    %aS = spatial.front(%S): memref<1024xi32>
    %aC = spatial.emplace(%C): memref<1024xi32>
    func.call @extern_kernel(%aA, %aB, %aS, %aC) : (memref<1024xi32>, memref<1024xi32>, memref<1024xi32>, memref<1024xi32>) -> ()
    spatial.pop(%A: !spatial.queue<memref<1024xi32>>)
    spatial.pop(%B: !spatial.queue<memref<1024xi32>>)
    spatial.pop(%S: !spatial.queue<memref<1024xi32>>)
    spatial.push(%C: !spatial.queue<memref<1024xi32>>)
    cf.br ^bb
  }

  %node_0_0 = spatial.node @kernel(%QA_0, %QB_0_0, %S_0_0, %S_0_1): (!spatial.queue<memref<1024xi32>>, !spatial.queue<memref<1024xi32>>, !spatial.queue<memref<1024xi32>>, !spatial.queue<memref<1024xi32>>) -> !spatial.node
  %node_0_1 = spatial.node @kernel(%QA_1, %QB_0_1, %S_0_1, %QC_0) : (!spatial.queue<memref<1024xi32>>, !spatial.queue<memref<1024xi32>>, !spatial.queue<memref<1024xi32>>, !spatial.queue<memref<1024xi32>>) -> !spatial.node
  %node_1_0 = spatial.node @kernel(%QA_0, %QB_1_0, %S_1_0, %S_1_1): (!spatial.queue<memref<1024xi32>>, !spatial.queue<memref<1024xi32>>, !spatial.queue<memref<1024xi32>>, !spatial.queue<memref<1024xi32>>) -> !spatial.node
  %node_1_1 = spatial.node @kernel(%QA_1, %QB_1_1, %S_1_1, %QC_1) : (!spatial.queue<memref<1024xi32>>, !spatial.queue<memref<1024xi32>>, !spatial.queue<memref<1024xi32>>, !spatial.queue<memref<1024xi32>>) -> !spatial.node

// COM: The checkings are directly adapted from PhysicalToAie/mm_2x2

// CHECK-DAG: physical.lock_acquire<1> (%[[ARG1:.*]])
// CHECK-NEXT: physical.lock_acquire<1> (%[[ARG2:.*]])
// CHECK-NEXT: physical.lock_acquire<1> (%[[ARG3:.*]])
// CHECK-NEXT: physical.lock_acquire<0> (%[[ARG4:.*]])
// CHECK-NEXT: call @extern_kernel
// CHECK-NEXT: physical.lock_release<0> (%[[ARG1:.*]])
// CHECK-NEXT: physical.lock_release<0> (%[[ARG2:.*]])
// CHECK-NEXT: physical.lock_release<0> (%[[ARG3:.*]])
// CHECK-NEXT: physical.lock_release<1> (%[[ARG4:.*]])
// CHECK-NEXT: cf.br

// CHECK-DAG:  physical.core @kernel{{.*}}(%[[A_0_a:.*]], %[[lA_0_a:.*]], %[[B_0_0:.*]], %[[lB_0_0:.*]], %[[S_0_0:.*]], %[[lS_0_0:.*]], %[[S_0_1:.*]], %[[lS_0_1:.*]]) {aie.tile = "6.3"}
// CHECK-DAG:  physical.core @kernel{{.*}}(%[[A_1_a:.*]], %[[lA_1_a:.*]], %[[B_0_1:.*]], %[[lB_0_1:.*]], %[[S_0_1]], %[[lS_0_1]], %[[C_0:.*]], %[[lC_0:.*]]) {aie.tile = "6.4"}
// CHECK-DAG:  physical.core @kernel{{.*}}(%[[A_0_b:.*]], %[[lA_0_b:.*]], %[[B_1_0:.*]], %[[lB_1_0:.*]], %[[S_1_0:.*]], %[[lS_1_0:.*]], %[[S_1_1:.*]], %[[lS_1_1:.*]]) {aie.tile = "7.3"}
// CHECK-DAG:  physical.core @kernel{{.*}}(%[[A_1_b:.*]], %[[lA_1_b:.*]], %[[B_1_1:.*]], %[[lB_1_1:.*]], %[[S_1_1]], %[[lS_1_1]], %[[C_1:.*]], %[[lC_1:.*]]) {aie.tile = "7.4"}

// CHECK-DAG: %[[leA_0:.*]]   = physical.lock<0> () {aie.id = "0", aie.tile = "6.0"}
// CHECK-DAG: %[[leA_1:.*]]   = physical.lock<0> () {aie.id = "1", aie.tile = "6.0"}
// CHECK-DAG: %[[leB_0_0:.*]] = physical.lock<0> () {aie.id = "2", aie.tile = "6.0"}
// CHECK-DAG: %[[leB_0_1:.*]] = physical.lock<0> () {aie.id = "3", aie.tile = "6.0"}
// CHECK-DAG: %[[leB_1_0:.*]] = physical.lock<0> () {aie.id = "0", aie.tile = "7.0"}
// CHECK-DAG: %[[leB_1_1:.*]] = physical.lock<0> () {aie.id = "1", aie.tile = "7.0"}
// CHECK-DAG: %[[leC_0:.*]]   = physical.lock<0> () {aie.id = "2", aie.tile = "7.0"}
// CHECK-DAG: %[[leC_1:.*]]   = physical.lock<0> () {aie.id = "3", aie.tile = "7.0"}
// CHECK-DAG: %[[lA_0_a]]  = physical.lock<0> () {{{.*}} aie.tile = "6.3"}
// CHECK-DAG: %[[lA_0_b]]  = physical.lock<0> () {{{.*}} aie.tile = "7.3"}
// CHECK-DAG: %[[lA_1_a]]  = physical.lock<0> () {{{.*}} aie.tile = "6.4"}
// CHECK-DAG: %[[lA_1_b]]  = physical.lock<0> () {{{.*}} aie.tile = "7.4"}
// CHECK-DAG: %[[lB_0_0]]  = physical.lock<0> () {{{.*}} aie.tile = "6.3"}
// CHECK-DAG: %[[lB_0_1]]  = physical.lock<0> () {{{.*}} aie.tile = "6.4"}
// CHECK-DAG: %[[lB_1_0]]  = physical.lock<0> () {{{.*}} aie.tile = "7.3"}
// CHECK-DAG: %[[lB_1_1]]  = physical.lock<0> () {{{.*}} aie.tile = "7.4"}
// CHECK-DAG: %[[lS_0_0]]  = physical.lock<0> () {{{.*}} aie.tile = "6.3"}
// CHECK-DAG: %[[lS_0_1]]  = physical.lock<0> () {{{.*}} aie.tile = "6.3"}
// CHECK-DAG: %[[lS_1_0]]  = physical.lock<0> () {{{.*}} aie.tile = "7.3"}
// CHECK-DAG: %[[lS_1_1]]  = physical.lock<0> () {{{.*}} aie.tile = "7.3"}
// CHECK-DAG: %[[lC_0]]    = physical.lock<0> () {{{.*}} aie.tile = "6.4"}
// CHECK-DAG: %[[lC_1]]    = physical.lock<0> () {{{.*}} aie.tile = "7.4"}

// CHECK-DAG: %[[eA_0:.*]]   = physical.buffer() {aie.external_address = "2203318222848"} : memref<1024xi32>
// CHECK-DAG: %[[eA_1:.*]]   = physical.buffer() {aie.external_address = "2203318226944"} : memref<1024xi32>
// CHECK-DAG: %[[eB_0_0:.*]] = physical.buffer() {aie.external_address = "2203318231040"} : memref<1024xi32>
// CHECK-DAG: %[[eB_0_1:.*]] = physical.buffer() {aie.external_address = "2203318235136"} : memref<1024xi32>
// CHECK-DAG: %[[eB_1_0:.*]] = physical.buffer() {aie.external_address = "2203318239232"} : memref<1024xi32>
// CHECK-DAG: %[[eB_1_1:.*]] = physical.buffer() {aie.external_address = "2203318243328"} : memref<1024xi32>
// CHECK-DAG: %[[eC_0:.*]]   = physical.buffer() {aie.external_address = "2203318247424"} : memref<1024xi32>
// CHECK-DAG: %[[eC_1:.*]]   = physical.buffer() {aie.external_address = "2203318251520"} : memref<1024xi32>
// CHECK-DAG: %[[A_0_a]]  = physical.buffer() {{{.*}} aie.tile = "6.3"} : memref<1024xi32>
// CHECK-DAG: %[[A_0_b]]  = physical.buffer() {{{.*}} aie.tile = "7.3"} : memref<1024xi32>
// CHECK-DAG: %[[A_1_a]]  = physical.buffer() {{{.*}} aie.tile = "6.4"} : memref<1024xi32>
// CHECK-DAG: %[[A_1_b]]  = physical.buffer() {{{.*}} aie.tile = "7.4"} : memref<1024xi32>
// CHECK-DAG: %[[B_0_0]]  = physical.buffer() {{{.*}} aie.tile = "6.3"} : memref<1024xi32>
// CHECK-DAG: %[[B_0_1]]  = physical.buffer() {{{.*}} aie.tile = "6.4"} : memref<1024xi32>
// CHECK-DAG: %[[B_1_0]]  = physical.buffer() {{{.*}} aie.tile = "7.3"} : memref<1024xi32>
// CHECK-DAG: %[[B_1_1]]  = physical.buffer() {{{.*}} aie.tile = "7.4"} : memref<1024xi32>
// CHECK-DAG: %[[S_0_0]]  = physical.buffer() {{{.*}} aie.tile = "6.3"} : memref<1024xi32>
// CHECK-DAG: %[[S_0_1]]  = physical.buffer() {{{.*}} aie.tile = "6.3"} : memref<1024xi32>
// CHECK-DAG: %[[S_1_0]]  = physical.buffer() {{{.*}} aie.tile = "7.3"} : memref<1024xi32>
// CHECK-DAG: %[[S_1_1]]  = physical.buffer() {{{.*}} aie.tile = "7.3"} : memref<1024xi32>
// CHECK-DAG: %[[C_0]]    = physical.buffer() {{{.*}} aie.tile = "6.4"} : memref<1024xi32>
// CHECK-DAG: %[[C_1]]    = physical.buffer() {{{.*}} aie.tile = "7.4"} : memref<1024xi32>

// CHECK-DAG: %[[sA1:.*]], %[[sA2:.*]]          = physical.stream<[[[T0:.*]], [[T1:.*]]]> () {aie.id = "0", aie.port = "DMA.O", aie.tile = "6.0"} : (!physical.ostream<i32>, !physical.istream<i32>)
// CHECK-DAG: %[[sA_0_a1:.*]], %[[sA_0_a2:.*]]  = physical.stream<[[[T0]]]>               () {aie.id = "0", aie.port = "DMA.I", aie.tile = "6.3"} : (!physical.ostream<i32>, !physical.istream<i32>)
// CHECK-DAG: %[[sA_0_b1:.*]], %[[sA_0_b2:.*]]  = physical.stream<[[[T0]]]>               () {aie.id = "0", aie.port = "DMA.I", aie.tile = "7.3"} : (!physical.ostream<i32>, !physical.istream<i32>)
// CHECK-DAG: %[[sA_1_a1:.*]], %[[sA_1_a2:.*]]  = physical.stream<[[[T1]]]>               () {aie.id = "0", aie.port = "DMA.I", aie.tile = "6.4"} : (!physical.ostream<i32>, !physical.istream<i32>)
// CHECK-DAG: %[[sA_1_b1:.*]], %[[sA_1_b2:.*]]  = physical.stream<[[[T1]]]>               () {aie.id = "0", aie.port = "DMA.I", aie.tile = "7.4"} : (!physical.ostream<i32>, !physical.istream<i32>)
// CHECK-DAG: %[[sB_01:.*]], %[[sB_02:.*]]      = physical.stream<[[[T2:.*]], [[T3:.*]]]> () {aie.id = "1", aie.port = "DMA.O", aie.tile = "6.0"} : (!physical.ostream<i32>, !physical.istream<i32>)
// CHECK-DAG: %[[sB_0_01:.*]], %[[sB_0_02:.*]]  = physical.stream<[[[T2]]]>               () {aie.id = "1", aie.port = "DMA.I", aie.tile = "6.3"} : (!physical.ostream<i32>, !physical.istream<i32>)
// CHECK-DAG: %[[sB_0_11:.*]], %[[sB_0_12:.*]]  = physical.stream<[[[T3]]]>               () {aie.id = "1", aie.port = "DMA.I", aie.tile = "6.4"} : (!physical.ostream<i32>, !physical.istream<i32>)
// CHECK-DAG: %[[sB_11:.*]], %[[sB_12:.*]]      = physical.stream<[[[T4:.*]], [[T5:.*]]]> () {aie.id = "0", aie.port = "DMA.O", aie.tile = "7.0"} : (!physical.ostream<i32>, !physical.istream<i32>)
// CHECK-DAG: %[[sB_1_01:.*]], %[[sB_1_02:.*]]  = physical.stream<[[[T4]]]>               () {aie.id = "1", aie.port = "DMA.I", aie.tile = "7.3"} : (!physical.ostream<i32>, !physical.istream<i32>)
// CHECK-DAG: %[[sB_1_11:.*]], %[[sB_1_12:.*]]  = physical.stream<[[[T5]]]>               () {aie.id = "1", aie.port = "DMA.I", aie.tile = "7.4"} : (!physical.ostream<i32>, !physical.istream<i32>)
// CHECK-DAG: %[[sC_01:.*]], %[[sC_02:.*]]      = physical.stream<[[[T6:.*]]]>            () {aie.id = "0", aie.port = "DMA.O", aie.tile = "6.4"} : (!physical.ostream<i32>, !physical.istream<i32>)
// CHECK-DAG: %[[seC_01:.*]], %[[seC_02:.*]]    = physical.stream<[[[T6]]]>               () {aie.id = "0", aie.port = "DMA.I", aie.tile = "7.0"} : (!physical.ostream<i32>, !physical.istream<i32>)
// CHECK-DAG: %[[sC_11:.*]], %[[sC_12:.*]]      = physical.stream<[[[T7:.*]]]>            () {aie.id = "0", aie.port = "DMA.O", aie.tile = "7.4"} : (!physical.ostream<i32>, !physical.istream<i32>)
// CHECK-DAG: %[[seC_11:.*]], %[[seC_12:.*]]    = physical.stream<[[[T7]]]>               () {aie.id = "1", aie.port = "DMA.I", aie.tile = "7.0"} : (!physical.ostream<i32>, !physical.istream<i32>)

// CHECK-DAG: physical.stream_hub(%[[sA2]], %[[sB_02]], %[[sB_12]], %[[sC_02]], %[[sC_12]], %[[sA_0_a1]], %[[sA_1_a1]], %[[sA_0_b1]], %[[sA_1_b1]], %[[sB_0_01]], %[[sB_0_11]], %[[sB_1_01]], %[[sB_1_11]], %[[seC_01]], %[[seC_11]]) {aie.impl = "broadcast_packet"}

// CHECK-DAG: physical.stream_dma(%[[sA1]] : !physical.ostream<i32>) {
// CHECK-DAG:   physical.stream_dma_connect<[[T0]]> (%[[leA_0]][1 -> 0], %[[eA_0]][0 : 1024] : memref<1024xi32>
// CHECK-DAG:   physical.stream_dma_connect<[[T1]]> (%[[leA_1]][1 -> 0], %[[eA_1]][0 : 1024] : memref<1024xi32>
// CHECK-DAG: } {aie.engine = "MM2S", aie.id = "0", aie.tile = "6.0"}
// CHECK-DAG: physical.stream_dma(%[[sA_0_a2]] : !physical.istream<i32>) {
// CHECK-DAG:   physical.stream_dma_connect (%[[lA_0_a]][0 -> 1], %[[A_0_a]][0 : 1024] : memref<1024xi32>
// CHECK-DAG: } {aie.engine = "S2MM", aie.id = "0", aie.tile = "6.3"}
// CHECK-DAG: physical.stream_dma(%[[sA_0_b2]] : !physical.istream<i32>) {
// CHECK-DAG:   physical.stream_dma_connect (%[[lA_0_b]][0 -> 1], %[[A_0_b]][0 : 1024] : memref<1024xi32>
// CHECK-DAG: } {aie.engine = "S2MM", aie.id = "0", aie.tile = "7.3"}
// CHECK-DAG: physical.stream_dma(%[[sA_1_a2]] : !physical.istream<i32>) {
// CHECK-DAG:   physical.stream_dma_connect (%[[lA_1_a]][0 -> 1], %[[A_1_a]][0 : 1024] : memref<1024xi32>
// CHECK-DAG: } {aie.engine = "S2MM", aie.id = "0", aie.tile = "6.4"}
// CHECK-DAG: physical.stream_dma(%[[sA_1_b2]] : !physical.istream<i32>) {
// CHECK-DAG:   physical.stream_dma_connect (%[[lA_1_b]][0 -> 1], %[[A_1_b]][0 : 1024] : memref<1024xi32>
// CHECK-DAG: } {aie.engine = "S2MM", aie.id = "0", aie.tile = "7.4"}
// CHECK-DAG: physical.stream_dma(%[[sB_01]] : !physical.ostream<i32>) {
// CHECK-DAG:   physical.stream_dma_connect<[[T2]]> (%[[leB_0_0]][1 -> 0], %[[eB_0_0]][0 : 1024] : memref<1024xi32>
// CHECK-DAG:   physical.stream_dma_connect<[[T3]]> (%[[leB_0_1]][1 -> 0], %[[eB_0_1]][0 : 1024] : memref<1024xi32>
// CHECK-DAG: } {aie.engine = "MM2S", aie.id = "1", aie.tile = "6.0"}
// CHECK-DAG: physical.stream_dma(%[[sB_11]] : !physical.ostream<i32>) {
// CHECK-DAG:   physical.stream_dma_connect<[[T4]]> (%[[leB_1_0]][1 -> 0], %[[eB_1_0]][0 : 1024] : memref<1024xi32>
// CHECK-DAG:   physical.stream_dma_connect<[[T5]]> (%[[leB_1_1]][1 -> 0], %[[eB_1_1]][0 : 1024] : memref<1024xi32>
// CHECK-DAG: } {aie.engine = "MM2S", aie.id = "0", aie.tile = "7.0"}
// CHECK-DAG: physical.stream_dma(%[[sB_0_02]] : !physical.istream<i32>) {
// CHECK-DAG:   physical.stream_dma_connect (%[[lB_0_0]][0 -> 1], %[[B_0_0]][0 : 1024] : memref<1024xi32>
// CHECK-DAG: } {aie.engine = "S2MM", aie.id = "1", aie.tile = "6.3"}
// CHECK-DAG: physical.stream_dma(%[[sB_0_12]] : !physical.istream<i32>) {
// CHECK-DAG:   physical.stream_dma_connect (%[[lB_0_1]][0 -> 1], %[[B_0_1]][0 : 1024] : memref<1024xi32>
// CHECK-DAG: } {aie.engine = "S2MM", aie.id = "1", aie.tile = "6.4"}
// CHECK-DAG: physical.stream_dma(%[[sB_1_02]] : !physical.istream<i32>) {
// CHECK-DAG:   physical.stream_dma_connect (%[[lB_1_0]][0 -> 1], %[[B_1_0]][0 : 1024] : memref<1024xi32>
// CHECK-DAG: } {aie.engine = "S2MM", aie.id = "1", aie.tile = "7.3"}
// CHECK-DAG: physical.stream_dma(%[[sB_1_12]] : !physical.istream<i32>) {
// CHECK-DAG:   physical.stream_dma_connect (%[[lB_1_1]][0 -> 1], %[[B_1_1]][0 : 1024] : memref<1024xi32>
// CHECK-DAG: } {aie.engine = "S2MM", aie.id = "1", aie.tile = "7.4"} 
// CHECK-DAG: physical.stream_dma(%[[sC_01]] : !physical.ostream<i32>) {
// CHECK-DAG:   physical.stream_dma_connect<[[T6]]> (%[[lC_0]][1 -> 0], %[[C_0]][0 : 1024] : memref<1024xi32>
// CHECK-DAG: } {aie.engine = "MM2S", aie.id = "0", aie.tile = "6.4"}
// CHECK-DAG: physical.stream_dma(%[[sC_11]] : !physical.ostream<i32>) {
// CHECK-DAG:   physical.stream_dma_connect<[[T7]]> (%[[lC_1]][1 -> 0], %[[C_1]][0 : 1024] : memref<1024xi32>
// CHECK-DAG: } {aie.engine = "MM2S", aie.id = "0", aie.tile = "7.4"}
// CHECK-DAG: physical.stream_dma(%[[seC_02]] : !physical.istream<i32>) {
// CHECK-DAG:   physical.stream_dma_connect (%[[leC_0]][0 -> 1], %[[eC_0]][0 : 1024] : memref<1024xi32>
// CHECK-DAG: } {aie.engine = "S2MM", aie.id = "0", aie.tile = "7.0"}
// CHECK-DAG: physical.stream_dma(%[[seC_12]] : !physical.istream<i32>) {
// CHECK-DAG:   physical.stream_dma_connect (%[[leC_1]][0 -> 1], %[[eC_1]][0 : 1024] : memref<1024xi32>
// CHECK-DAG: } {aie.engine = "S2MM", aie.id = "1", aie.tile = "7.0"}

  layout.platform<"vck190"> {
    layout.device<"aie"> {

      // TODO: move to layout.device<"global_memory">
      layout.place<"external_address/2203318222848/buffer,tile/6.0/id/0/lock">(%QA_0: !spatial.queue<memref<1024xi32>>)
      layout.place<"external_address/2203318226944/buffer,tile/6.0/id/1/lock">(%QA_1: !spatial.queue<memref<1024xi32>>)
      layout.place<"external_address/2203318231040/buffer,tile/6.0/id/2/lock">(%QB_0_0: !spatial.queue<memref<1024xi32>>)
      layout.place<"external_address/2203318235136/buffer,tile/6.0/id/3/lock">(%QB_0_1: !spatial.queue<memref<1024xi32>>)
      layout.place<"external_address/2203318239232/buffer,tile/7.0/id/0/lock">(%QB_1_0: !spatial.queue<memref<1024xi32>>)
      layout.place<"external_address/2203318243328/buffer,tile/7.0/id/1/lock">(%QB_1_1: !spatial.queue<memref<1024xi32>>)
      layout.place<"external_address/2203318247424/buffer,tile/7.0/id/2/lock">(%QC_0: !spatial.queue<memref<1024xi32>>)
      layout.place<"external_address/2203318251520/buffer,tile/7.0/id/3/lock">(%QC_1: !spatial.queue<memref<1024xi32>>)
      
      layout.place<"tile/6.3/id/2/buffer,tile/6.3/id/2/lock">(%S_0_0: !spatial.queue<memref<1024xi32>>)
      layout.place<"tile/6.3/id/3/buffer,tile/6.3/id/3/lock">(%S_0_1: !spatial.queue<memref<1024xi32>>)
      layout.place<"tile/7.3/id/2/buffer,tile/7.3/id/2/lock">(%S_1_0: !spatial.queue<memref<1024xi32>>)
      layout.place<"tile/7.3/id/3/buffer,tile/7.3/id/3/lock">(%S_1_1: !spatial.queue<memref<1024xi32>>)
      layout.place<"tile/6.3/core">(%node_0_0: !spatial.node)
      layout.place<"tile/6.4/core">(%node_0_1: !spatial.node)
      layout.place<"tile/7.3/core">(%node_1_0: !spatial.node)
      layout.place<"tile/7.4/core">(%node_1_1: !spatial.node)

      // Direct accesses
      layout.route<[]>(%S_0_0: !spatial.queue<memref<1024xi32>> -> %node_0_0: !spatial.node)
      layout.route<[]>(%node_0_0: !spatial.node -> %S_0_1: !spatial.queue<memref<1024xi32>>)
      layout.route<[]>(%S_0_1: !spatial.queue<memref<1024xi32>> -> %node_0_1: !spatial.node)
      
      // Direct accesses
      layout.route<[]>(%S_1_0: !spatial.queue<memref<1024xi32>> -> %node_1_0: !spatial.node)
      layout.route<[]>(%node_1_0: !spatial.node -> %S_1_1: !spatial.queue<memref<1024xi32>>)
      layout.route<[]>(%S_1_1: !spatial.queue<memref<1024xi32>> -> %node_1_1: !spatial.node)

      layout.route<["tile/6.0/engine/MM2S/id/0/stream_dma", "tile/6.0/port/DMA.O/id/0/stream", "impl/broadcast_packet/stream_hub",
                    "tile/6.3/port/DMA.I/id/0/stream", "tile/6.3/engine/S2MM/id/0/stream_dma", "tile/6.3/id/0/buffer,tile/6.3/id/0/lock"]>
                    (%QA_0: !spatial.queue<memref<1024xi32>> -> %node_0_0: !spatial.node)
      layout.route<["tile/6.0/engine/MM2S/id/0/stream_dma", "tile/6.0/port/DMA.O/id/0/stream", "impl/broadcast_packet/stream_hub",
                    "tile/6.4/port/DMA.I/id/0/stream", "tile/6.4/engine/S2MM/id/0/stream_dma", "tile/6.4/id/0/buffer,tile/6.4/id/0/lock"]>
                    (%QA_1: !spatial.queue<memref<1024xi32>> -> %node_0_1: !spatial.node)
      layout.route<["tile/6.0/engine/MM2S/id/0/stream_dma", "tile/6.0/port/DMA.O/id/0/stream", "impl/broadcast_packet/stream_hub",
                    "tile/7.3/port/DMA.I/id/0/stream", "tile/7.3/engine/S2MM/id/0/stream_dma", "tile/7.3/id/0/buffer,tile/7.3/id/0/lock"]>
                    (%QA_0: !spatial.queue<memref<1024xi32>> -> %node_1_0: !spatial.node)
      layout.route<["tile/6.0/engine/MM2S/id/0/stream_dma", "tile/6.0/port/DMA.O/id/0/stream", "impl/broadcast_packet/stream_hub",
                    "tile/7.4/port/DMA.I/id/0/stream", "tile/7.4/engine/S2MM/id/0/stream_dma", "tile/7.4/id/0/buffer,tile/7.4/id/0/lock"]>
                    (%QA_1: !spatial.queue<memref<1024xi32>> -> %node_1_1: !spatial.node)

      layout.route<["tile/6.0/engine/MM2S/id/1/stream_dma", "tile/6.0/port/DMA.O/id/1/stream", "impl/broadcast_packet/stream_hub",
                    "tile/6.3/port/DMA.I/id/1/stream", "tile/6.3/engine/S2MM/id/1/stream_dma", "tile/6.3/id/1/buffer,tile/6.3/id/1/lock"]>
                    (%QB_0_0: !spatial.queue<memref<1024xi32>> -> %node_0_0: !spatial.node)
      layout.route<["tile/6.0/engine/MM2S/id/1/stream_dma", "tile/6.0/port/DMA.O/id/1/stream", "impl/broadcast_packet/stream_hub",
                    "tile/6.4/port/DMA.I/id/1/stream", "tile/6.4/engine/S2MM/id/1/stream_dma", "tile/6.4/id/1/buffer,tile/6.4/id/1/lock"]>
                    (%QB_0_1: !spatial.queue<memref<1024xi32>> -> %node_0_1: !spatial.node)

      layout.route<["tile/7.0/engine/MM2S/id/0/stream_dma", "tile/7.0/port/DMA.O/id/0/stream", "impl/broadcast_packet/stream_hub",
                    "tile/7.3/port/DMA.I/id/1/stream", "tile/7.3/engine/S2MM/id/1/stream_dma", "tile/7.3/id/1/buffer,tile/7.3/id/1/lock"]>
                    (%QB_1_0: !spatial.queue<memref<1024xi32>> -> %node_1_0: !spatial.node)
      layout.route<["tile/7.0/engine/MM2S/id/0/stream_dma", "tile/7.0/port/DMA.O/id/0/stream", "impl/broadcast_packet/stream_hub",
                    "tile/7.4/port/DMA.I/id/1/stream", "tile/7.4/engine/S2MM/id/1/stream_dma", "tile/7.4/id/1/buffer,tile/7.4/id/1/lock"]>
                    (%QB_1_1: !spatial.queue<memref<1024xi32>> -> %node_1_1: !spatial.node)

      layout.route<["tile/6.4/id/2/buffer,tile/6.4/id/2/lock", "tile/6.4/engine/MM2S/id/0/stream_dma", "tile/6.4/port/DMA.O/id/0/stream",
                    "impl/broadcast_packet/stream_hub", "tile/7.0/port/DMA.I/id/0/stream", "tile/7.0/engine/S2MM/id/0/stream_dma"]>
                    (%node_0_1: !spatial.node -> %QC_0: !spatial.queue<memref<1024xi32>>)
      layout.route<["tile/7.4/id/2/buffer,tile/7.4/id/2/lock", "tile/7.4/engine/MM2S/id/0/stream_dma", "tile/7.4/port/DMA.O/id/0/stream",
                    "impl/broadcast_packet/stream_hub", "tile/7.0/port/DMA.I/id/1/stream", "tile/7.0/engine/S2MM/id/1/stream_dma"]>
                    (%node_1_1: !spatial.node -> %QC_1: !spatial.queue<memref<1024xi32>>)
    }
  }
}
