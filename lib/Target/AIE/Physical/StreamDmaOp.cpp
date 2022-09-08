//===- StreamDmaOp.cpp ------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "phy/Target/AIE/Physical/StreamDmaOp.h"

#include "phy/Dialect/Physical/PhysicalDialect.h"
#include "phy/Target/AIE/TargetResources.h"

#include <list>
#include <map>

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace phy::physical;
using namespace phy::target::aie;
using namespace xilinx;

class StreamDmaOpToAieLowering : public OpConversionPattern<StreamDmaOp> {
  AIELoweringPatternSets *lowering;
  using OpAdaptor = typename StreamDmaOp::Adaptor;

public:
  StreamDmaOpToAieLowering(mlir::MLIRContext *context,
                           AIELoweringPatternSets *lowering)
      : OpConversionPattern<StreamDmaOp>(context), lowering(lowering) {}

  mlir::LogicalResult
  matchAndRewrite(StreamDmaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    mlir::LogicalResult result = success();

    auto tile = lowering->getTileIndex(op);
    auto &connections = op.connections().front();

    if (TargetResources().isShimTile(tile.first, tile.second)) {
      result = rewriteBlock(op, &connections, lowering->getShimDma(tile));
    } else {
      result = rewriteBlock(op, &connections, lowering->getDma(tile));
    }

    if (result.succeeded())
      rewriter.eraseOp(op);
    return result;
  }

  template <typename DMAOp>
  mlir::LogicalResult rewriteBlock(StreamDmaOp op, mlir::Block *connections,
                                   DMAOp dma) const {

    std::map<StreamDmaConnectOp, mlir::Block *> connect_bd_blocks;
    std::list<mlir::Block *> bd_blocks;

    auto &last_bd = dma.body().front();
    auto &aie_end = dma.body().back();
    auto stream = dyn_cast_or_null<StreamOp>(op.endpoint().getDefiningOp());
    assert(stream &&
           "stream dma must directly refer to the stream definition.");

    // Create DMA BD blocks
    for (auto &connect_op : *connections) {
      if (isa<EndOp>(connect_op))
        continue;
      auto connect = dyn_cast_or_null<StreamDmaConnectOp>(connect_op);
      assert(connect && "stream dma can only contain StreamDmaConnectOp.");

      auto bd_block = new mlir::Block();
      bd_blocks.push_back(bd_block);
      connect_bd_blocks[connect] = bd_block;
    }

    // Push DMA BD blocks to the front in the order
    for (auto it = bd_blocks.rbegin(); it != bd_blocks.rend(); it++) {
      dma.body().push_front(*it);
    }

    // Construct the DMA BD block
    for (auto bd_block : connect_bd_blocks) {
      constructDMABDBlock(bd_block.first, bd_block.second, &aie_end,
                          connect_bd_blocks);
    }

    // Prepend and chain BDs
    // AIE.dmaStart("${engine/port}${id}", ^first_block, ^last_bd)
    auto chain_block = new mlir::Block();
    dma.body().push_front(chain_block);
    auto builder = OpBuilder::atBlockBegin(chain_block);
    builder.create<AIE::DMAStartOp>(builder.getUnknownLoc(),
                                    lowering->getChannel(op, stream),
                                    bd_blocks.front(), &last_bd);

    return success();
  }

  void constructDMABDBlock(
      StreamDmaConnectOp connect, mlir::Block *bd_block, mlir::Block *aie_end,
      std::map<StreamDmaConnectOp, mlir::Block *> &connect_bd_blocks) const {

    auto builder = OpBuilder::atBlockBegin(bd_block);

    auto phy_lock = dyn_cast_or_null<LockOp>(connect.lock().getDefiningOp());
    assert(phy_lock &&
           "stream dma must directly refer to the lock definition.");

    auto tile = lowering->getTile(phy_lock);
    auto id = lowering->getId(phy_lock);
    auto lock = lowering->getLock(tile, id);

    // AIE.useLock(%lock, Acquire, acquire())
    builder.create<AIE::UseLockOp>(builder.getUnknownLoc(), lock,
                                   connect.acquire(), AIE::LockAction::Acquire);

    // AIE.dmaBdPacket(tag(), tag())
    if (connect.tag().hasValue()) {
      builder.create<AIE::DMABDPACKETOp>(
          builder.getUnknownLoc(),
          builder.getI32IntegerAttr(connect.tag().getValue()),
          builder.getI32IntegerAttr(connect.tag().getValue()));
    }

    // AIE.dmaBd(<%buffer, start(), end() - start()>, 0)
    builder.create<AIE::DMABDOp>(
        builder.getUnknownLoc(), connect.buffer(),
        builder.getI32IntegerAttr(connect.start()),
        builder.getI32IntegerAttr(connect.end() - connect.start()),
        builder.getI32IntegerAttr(0));

    // AIE.useLock(%lock, Release, release())
    builder.create<AIE::UseLockOp>(builder.getUnknownLoc(), lock,
                                   connect.release(), AIE::LockAction::Release);

    // cf.br ^next
    auto next_bd_block = aie_end;
    if (connect.next()) {
      auto next_op = connect.next().getDefiningOp();
      auto next_connect = dyn_cast<StreamDmaConnectOp>(next_op);
      next_bd_block = connect_bd_blocks[next_connect];
    }
    builder.create<cf::BranchOp>(builder.getUnknownLoc(), next_bd_block);
  }
};

void StreamDmaOpLoweringPatternSet::populatePatternSet(
    mlir::RewritePatternSet &patterns) {

  patterns.add<StreamDmaOpToAieLowering>(patterns.getContext(), lowering);
}

void StreamDmaOpLoweringPatternSet::populateTarget(
    mlir::ConversionTarget &target) {
  target.addLegalOp<cf::BranchOp>();
}
