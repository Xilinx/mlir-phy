# mlir-phy
⏩ An MLIR dialect to express the mapping of PEs, buffers, and buses to an abstract physical floorplan.

## Definition

- **Abstraction Level: [TAPA](https://github.com/UCLA-VAST/tapa) that supports multiple targets when floorplanning is involved, e.g., AutoBridge and RapidStream.**
- This dialect defines flattened free-running processing elements (PEs), buffers (memory that stores data), and buses (streams or packets that transmit data) with spatial information (how are they connected and how many of them are implemented).  The elements defined in this dialect can be mapped to the physical regions of the devices.
- A PE can have a device-specific attribute of location, e.g. a thread in CPU, a tile in AIE, or a reconfiguration slot in FPGA.
- A buffer can have only one location but its data can be pushed to another buffer initiated in a PE when connected with a bus.
  - A buffer can be randomly accessed by a PE only when they are at the exact location or supported by the platform.
  - A buffer can be sequentially accessed by a PE using a DMA bus even if they are not at the exact location.
- A bus connects multiple PEs and buffers for message-passing, DMAs, and memcpy-ing. It can have a device-specific attribute of the route.

**WARNING:** This is a work-in-progress and will be actively changed.

## Sample APIs

```mlir
func.func @func(%mem: phy.buffer<1024xi32>, %bus: phy.bus<i32>) {
  %b1 = phy.non_empty(%bus, BusID)   // non-blocking test if able to receive
  %b2 = phy.non_full(%bus, BusID)    // non-blocking test if able to send
  %v1 = phy.pop(%bus, BusID)         // blocking receive via bus
  phy.push(%v1, %bus, BusID)         // blocking send via bus

  phy.acquire_buffer(%mem, OwnerID)
  // blocking acquire the buffer for read or write as OwnerID
  phy.release_buffer(%mem, OtherID)
  // release the buffer so that it can be used by OtherID
  %v2 = phy.load(%mem[%idx])
  phy.store(%v2, %mem[%idx])
}

%bus = phy.bus(type, minDepth)
%buffer = phy.buffer(type, minSize, initOwnerID) {
  in_bus  = [(%bus1, BusID, OwnerID, OtherID), ...]
  // the buffer will be written to using bus1.BusID when released for OwnerID,
  // and it will then be released for OtherID when the writing is finished
  out_bus = [(%bus2, BusID, OwnerID, OtherID), ...]|
  // the buffer's content will be sent to bus2.BusID
}

%pe = phy.pe @func(%buffer, %buffer2, %bus)
// PEs will be started free-running once the system is up

phy.platform('platformName') {
  phy.device(%plat, 'deviceName') {
    phy.place(%pe/%buffer, 'locationName') 
    phy.route(%bus, ['wireName', ...])
  }
  phy.route(%bus, ['PLIO', ...])
}
phy.route(%bus, ['PCIe', ...])
```

## Discussion

### Programmability

- This dialect acts as an assembly for hardware that involves floorplanning fine-tuning.
- Fancy shared-memory and message-passing protocols are lowered as buffers and buses.
- A buffer has a lock integrated as different devices have different implementations for this combination.
- A bus acts as stream and packet switching to simplify routing modeling.

### Expressiveness

- Expresses hardware implementation where each PE, buffer, and bus must be placed in a physical location.
- Everything is static from an overall architectural perspective, other than `OwnerID` and `BusID` for multiplexing.
- Expressiveness in code `func.func` is target-dependent.

### Compiler's Degrees of Freedom

- Compiler optimizations could move the location/routes around using heuristics.
- External tools estimate the latency and throughput when a PE is placed in a region.
- A bus occupied the circuit (circuit switching when point to point), so the latency and throughput per transaction can be determined.
- Overall throughput or latency can be optimized based on estimations.
