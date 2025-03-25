# Vector-Based Forwarding (VBF) Simulation

This project contains simulations of the Vector-Based Forwarding (VBF) protocol for Underwater Sensor Networks (UWSNs) using matplotlib.

## What is VBF?

Vector-Based Forwarding (VBF) is a routing protocol for Underwater Sensor Networks that works as follows:

1. **Vector Formation:**
   - A source node sends a packet with a vector from itself to the destination (sink).
   - Nodes near this vector form a "routing pipe", filtering unnecessary nodes.

2. **Forwarding Strategy:**
   - Only nodes inside the routing pipe participate in forwarding.
   - The closest node to the vector forwards the packet.
   - This process continues until the sink node receives the packet.

3. **Adaptive Decision:**
   - If no suitable node exists inside the pipe, the packet is dropped or retransmitted.
   - Nodes can adjust the pipe's radius dynamically based on network density.

## Files

- `vbf_simulation.py`: Basic 2D visualization of VBF
- `vbf_simple.py`: Enhanced simulation with both static and animated visualizations

## Requirements

The simulations require the following Python packages:
- numpy
- matplotlib
- scipy (for advanced calculations)

All dependencies are listed in the `requirements.txt` file.

## Running the Simulations

### Basic Simulation

```bash
python vbf_simulation.py
```

This will generate a static 2D visualization of the VBF routing protocol with:
- A blue routing pipe between source and sink
- Gray nodes representing all sensor nodes
- Blue nodes representing the nodes inside the routing pipe
- A red path showing the packet forwarding route

### Enhanced Simulation

```bash
python vbf_simple.py
```

This will run:
1. A static visualization of the complete routing path
2. An animated visualization showing the step-by-step packet forwarding process

## Customization

You can customize the simulations by modifying the following parameters:
- `num_nodes`: Number of sensor nodes
- `pipe_radius`: Radius of the routing pipe
- Source and sink node positions

## References

VBF was introduced in:
- P. Xie, J. H. Cui, and L. Lao, "VBF: Vector-Based Forwarding Protocol for Underwater Sensor Networks," in Proceedings of IFIP Networking 2006. 