import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path
import matplotlib.animation as animation
from matplotlib.backends.backend_agg import FigureCanvasAgg
from io import BytesIO
import base64
import time
import pandas as pd
import importlib
from PIL import Image
import io

# Use imageio v2 to avoid deprecation warnings
import imageio.v2 as imageio

# Import the protocol implementations
from vbf_simulation import VBFSimulation
from focused_beam_routing import FocusedBeamRouter, Node, Packet
from uwsn_clustering import UWSNClusteringProtocol

# Function to create animations in memory without saving files
def create_animation_from_figures(figures, fps=1):
    """Create an animation from a list of matplotlib figures without saving to disk"""
    # Convert figures to images in memory
    images = []
    
    for fig in figures:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        images.append(img)
        plt.close(fig)
    
    # Create an in-memory GIF
    gif_buf = io.BytesIO()
    images[0].save(
        gif_buf, 
        format='GIF',
        append_images=images[1:],
        save_all=True,
        duration=1000//fps,  # ms between frames
        loop=0                # 0 means loop forever
    )
    gif_buf.seek(0)
    
    return gif_buf

# Define a function to show figures in Streamlit
def show_figure(fig):
    """Display a matplotlib figure in Streamlit"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    st.image(buf)
    plt.close(fig)

# Check if all modules are available
required_modules = ['streamlit', 'numpy', 'matplotlib', 'imageio', 'pandas', 'PIL']
missing_modules = []
for module in required_modules:
    try:
        importlib.import_module(module)
    except ImportError:
        missing_modules.append(module)

if missing_modules:
    st.error(f"Missing required modules: {', '.join(missing_modules)}")
    st.info("Please install them using: pip install " + " ".join(missing_modules))
    st.stop()

st.set_page_config(layout="wide", page_title="UWSN Protocol Simulator")

# Define the app title and introduction
st.title("Underwater Sensor Network Protocol Simulator")
st.markdown("""
This application allows you to simulate and compare three different routing protocols for Underwater Sensor Networks (UWSNs):
- **Vector-Based Forwarding (VBF)** - Creates a routing pipe between source and destination
- **Focused Beam Routing (FBR)** - Uses directed beams to route packets
- **UWSN Clustering** - Forms node clusters for efficient data aggregation and transmission
""")

# Create tabs for each protocol
tab1, tab2, tab3 = st.tabs(["Vector-Based Forwarding", "Focused Beam Routing", "UWSN Clustering"])

# Function to generate an animated GIF from a matplotlib animation
def anim_to_gif(anim, filename="animation.gif", fps=2):
    filepath = f"{filename}"
    anim.save(filepath, writer='pillow', fps=fps)
    return filepath

# Function to convert figure to HTML
def fig_to_html(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f'<img src="data:image/png;base64,{data}"/>'

#--------------------- Vector-Based Forwarding (VBF) Tab ---------------------#
with tab1:
    st.header("Vector-Based Forwarding (VBF) Protocol")
    st.markdown("""
    VBF is a routing protocol where only nodes within a virtual "pipe" from source to destination participate in data forwarding.
    This reduces energy consumption and network traffic by involving fewer nodes in the routing process.
    """)
    
    # Input parameters for VBF
    st.subheader("Simulation Parameters")
    vbf_col1, vbf_col2, vbf_col3 = st.columns(3)
    
    with vbf_col1:
        vbf_width = st.slider("Simulation Area Width", 500, 2000, 1000, step=100, key="vbf_width")
        vbf_height = st.slider("Simulation Area Height", 500, 2000, 1000, step=100, key="vbf_height")
    
    with vbf_col2:
        vbf_num_nodes = st.slider("Number of Nodes", 50, 300, 150, step=10, key="vbf_num_nodes")
        vbf_pipe_radius = st.slider("Pipe Radius", 50, 300, 120, step=10, key="vbf_pipe_radius")
    
    with vbf_col3:
        vbf_source_x = st.slider("Source X Position", 0, vbf_width, 100, step=10, key="vbf_source_x")
        vbf_source_y = st.slider("Source Y Position", 0, vbf_height, int(vbf_height/2), step=10, key="vbf_source_y")
        vbf_sink_x = st.slider("Sink X Position", 0, vbf_width, 900, step=10, key="vbf_sink_x")
        vbf_sink_y = st.slider("Sink Y Position", 0, vbf_height, int(vbf_height/2), step=10, key="vbf_sink_y")
    
    # Button to run VBF simulation
    if st.button("Run VBF Simulation", key="run_vbf"):
        with st.spinner("Running VBF simulation..."):
            # Create a custom VBF simulation class with specified parameters
            class CustomVBFSimulation(VBFSimulation):
                def __init__(self, width, height, num_nodes, pipe_radius, source_pos, sink_pos):
                    self.width = width
                    self.height = height
                    self.num_nodes = num_nodes
                    self.pipe_radius = pipe_radius
                    
                    # Set source and sink positions
                    self.source = np.array(source_pos)
                    self.sink = np.array(sink_pos)
                    
                    # Generate random node positions
                    np.random.seed(42)  # For reproducibility
                    self.nodes = np.random.rand(num_nodes, 2)
                    self.nodes[:, 0] = self.nodes[:, 0] * width
                    self.nodes[:, 1] = self.nodes[:, 1] * height
                    
                    # Vector from source to sink
                    self.routing_vector = self.sink - self.source
                    self.unit_vector = self.routing_vector / np.linalg.norm(self.routing_vector)
                    
                    # Identify nodes in the routing pipe
                    self.pipe_nodes_indices = self._find_pipe_nodes()
                    self.pipe_nodes = self.nodes[self.pipe_nodes_indices]
                    
                    # Find forwarding path
                    self.path = self._calculate_forwarding_path()
                    
                    # Performance metrics
                    self.energy_consumption = 0
                    self.transmission_delays = []
                    self.total_delay = 0
                    self.packet_delivery_ratio = 0
                    self.hop_count = len(self.path) - 1
                    
                    # Calculate performance metrics
                    self._calculate_performance_metrics()
            
            # Create the simulation
            vbf_sim = CustomVBFSimulation(
                width=vbf_width,
                height=vbf_height,
                num_nodes=vbf_num_nodes,
                pipe_radius=vbf_pipe_radius,
                source_pos=[vbf_source_x, vbf_source_y],
                sink_pos=[vbf_sink_x, vbf_sink_y]
            )
            
            # Create the static visualization
            st.subheader("VBF Network Visualization")
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Set up the plot area
            ax.set_xlim(0, vbf_sim.width)
            ax.set_ylim(0, vbf_sim.height)
            ax.set_title('Vector-Based Forwarding (VBF) Protocol Simulation')
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            
            # Draw the routing pipe
            vbf_sim._draw_routing_pipe(ax)
            
            # Plot all nodes
            ax.scatter(vbf_sim.nodes[:, 0], vbf_sim.nodes[:, 1], c='gray', alpha=0.5, label='Sensor Nodes')
            
            # Highlight pipe nodes
            ax.scatter(vbf_sim.pipe_nodes[:, 0], vbf_sim.pipe_nodes[:, 1], c='blue', s=80, label='Pipe Nodes')
            
            # Plot source and sink
            ax.scatter(vbf_sim.source[0], vbf_sim.source[1], c='green', s=150, marker='*', label='Source')
            ax.scatter(vbf_sim.sink[0], vbf_sim.sink[1], c='purple', s=150, marker='*', label='Sink')
            
            # Plot the forwarding path
            ax.plot(vbf_sim.path[:, 0], vbf_sim.path[:, 1], 'r-', lw=2, label='Forwarding Path')
            
            # Mark each hop in the path
            ax.scatter(vbf_sim.path[1:-1, 0], vbf_sim.path[1:-1, 1], c='red', s=120, marker='o', edgecolors='k', label='Forwarding Nodes')
            
            ax.legend()
            plt.tight_layout()
            
            # Show the static visualization
            show_figure(fig)
            
            # Display performance metrics
            st.subheader("VBF Performance Metrics")
            
            vbf_metric_col1, vbf_metric_col2, vbf_metric_col3 = st.columns(3)
            
            with vbf_metric_col1:
                st.metric("Hop Count", vbf_sim.hop_count)
                st.metric("Nodes in Pipe", f"{len(vbf_sim.pipe_nodes_indices)} ({len(vbf_sim.pipe_nodes_indices)/vbf_sim.num_nodes*100:.1f}%)")
                
            with vbf_metric_col2:
                st.metric("Energy Consumption", f"{vbf_sim.energy_consumption:.2f} μJ")
                st.metric("Energy per Hop", f"{vbf_sim.energy_consumption/max(1, vbf_sim.hop_count):.2f} μJ/hop")
                
            with vbf_metric_col3:
                st.metric("End-to-End Delay", f"{vbf_sim.total_delay:.4f} s")
                st.metric("Packet Delivery Ratio", f"{vbf_sim.packet_delivery_ratio:.4f}")
            
            # Plot performance metrics
            st.subheader("VBF Performance Analysis")
            
            # Generate and show performance metrics visualization
            vbf_sim.plot_performance_metrics()
            
            # Show animation in streamlit (saved as gif)
            st.subheader("VBF Data Transmission Animation")
            
            # Create animation frames in memory instead of saving to disk
            animation_frames = []
            
            for i in range(len(vbf_sim.path)):
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Set up the plot area
                ax.set_xlim(0, vbf_sim.width)
                ax.set_ylim(0, vbf_sim.height)
                
                # Draw routing pipe
                vbf_sim._draw_routing_pipe(ax)
                
                # Plot all nodes
                ax.scatter(vbf_sim.nodes[:, 0], vbf_sim.nodes[:, 1], c='gray', alpha=0.5, label='Sensor Nodes')
                
                # Highlight pipe nodes
                ax.scatter(vbf_sim.pipe_nodes[:, 0], vbf_sim.pipe_nodes[:, 1], c='blue', s=80, label='Pipe Nodes')
                
                # Plot source and sink
                ax.scatter(vbf_sim.source[0], vbf_sim.source[1], c='green', s=150, marker='*', label='Source')
                ax.scatter(vbf_sim.sink[0], vbf_sim.sink[1], c='purple', s=150, marker='*', label='Sink')
                
                # Plot the partial path up to current position
                ax.plot(vbf_sim.path[:i+1, 0], vbf_sim.path[:i+1, 1], 'r-', lw=2, label='Forwarding Path')
                
                # Add current position marker
                ax.scatter(vbf_sim.path[i, 0], vbf_sim.path[i, 1], c='yellow', s=180, marker='o', 
                          edgecolors='red', linewidth=2, zorder=10, label='Packet Position')
                
                # Add title based on position
                if i == 0:
                    ax.set_title("Data packet generated at source", fontsize=14)
                elif i == len(vbf_sim.path) - 1:
                    ax.set_title("Data packet reached destination!", fontsize=14)
                else:
                    hop_num = i
                    ax.set_title(f"Hop #{hop_num}: Moving toward destination", fontsize=14)
                
                # Add progress indicator
                progress = i / (len(vbf_sim.path) - 1) if len(vbf_sim.path) > 1 else 0
                plt.figtext(0.5, 0.01, f"Progress: {progress*100:.1f}%", 
                         ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
                
                ax.legend(loc='upper right')
                ax.grid(True, alpha=0.3)
                
                # Add to frames instead of saving to disk
                animation_frames.append(fig)
            
            # Display the animation
            try:
                if animation_frames:
                    # Generate animation in memory
                    animation_data = create_animation_from_figures(animation_frames, fps=1)
                    
                    # Display animation
                    st.subheader("VBF Protocol Animation")
                    st.image(animation_data)
                else:
                    st.warning("No animation frames were generated.")
            except Exception as e:
                st.error(f"Animation error: {str(e)}. Showing static visualization instead.")
                # Show static visualization as fallback
                fig = plt.figure(figsize=(10, 8))
                vbf_sim.visualize_static()
                show_figure(plt.gcf())

#--------------------- Focused Beam Routing (FBR) Tab ---------------------#
with tab2:
    st.header("Focused Beam Routing (FBR) Protocol")
    st.markdown("""
    Focused Beam Routing (FBR) is a position-based routing protocol for UWSNs that uses directional beams 
    from the source to guide the packet along a path to the destination. The protocol adjusts the beam angle to 
    find an optimal path while reducing energy consumption.
    """)
    
    # Input parameters for FBR
    st.subheader("Simulation Parameters")
    fbr_col1, fbr_col2, fbr_col3 = st.columns(3)
    
    with fbr_col1:
        fbr_num_nodes = st.slider("Number of Nodes", 20, 100, 50, step=5, key="fbr_num_nodes")
        fbr_world_size = st.slider("World Size", 500, 2000, 1000, step=100, key="fbr_world_size")
    
    with fbr_col2:
        fbr_max_range = st.slider("Node Max Range", 100, 500, 350, step=50, key="fbr_max_range")
        fbr_beam_width = st.slider("Beam Width (degrees)", 10, 90, 45, step=5, key="fbr_beam_width")
    
    with fbr_col3:
        fbr_energy = st.slider("Initial Node Energy", 50, 200, 100, step=10, key="fbr_energy")
        fbr_power_levels = st.slider("Power Levels", 1, 5, 3, step=1, key="fbr_power_levels")
    
    # Button to run FBR simulation
    if st.button("Run FBR Simulation", key="run_fbr"):
        with st.spinner("Running FBR simulation..."):
            # Create a new FBR router with the specified parameters
            router = FocusedBeamRouter()
            
            # Create nodes with specified parameters
            np.random.seed(42)  # For reproducibility
            
            # Create source and sink nodes at specific positions
            source_pos = np.array([fbr_world_size * 0.1, fbr_world_size * 0.5, 0])
            sink_pos = np.array([fbr_world_size * 0.9, fbr_world_size * 0.5, 0])
            
            # Add source node (id=0)
            source_node = Node(0, source_pos, max_range=fbr_max_range, energy=fbr_energy)
            router.add_node(source_node)
            
            # Add sink node (id=1)
            sink_node = Node(1, sink_pos, max_range=fbr_max_range, energy=fbr_energy)
            router.add_node(sink_node)
            
            # Add nodes along the corridor
            for i in range(2, fbr_num_nodes):
                # Create nodes with a better distribution for successful routing
                if i < fbr_num_nodes * 0.8:  # Place 80% of nodes along the corridor
                    # Use a more structured approach with interval-based placement
                    segment_count = (fbr_num_nodes * 0.8)
                    segment_index = (i - 2) % segment_count
                    progress = segment_index / segment_count
                    
                    # Create a corridor that's wider near endpoints and narrower in the middle
                    normalized_progress = 2 * abs(progress - 0.5)  # 0 at midpoint, 1 at endpoints
                    corridor_width = fbr_world_size * (0.2 + 0.2 * normalized_progress)
                    
                    # Position along the corridor
                    pos_x = source_pos[0] + progress * (sink_pos[0] - source_pos[0])
                    pos_y = source_pos[1] + progress * (sink_pos[1] - source_pos[1])
                    
                    # Add alternating deviation from center path
                    deviation_direction = 1 if i % 2 == 0 else -1
                    deviation_magnitude = np.random.uniform(0.1, 1.0) * corridor_width/2 * deviation_direction
                    pos_y += deviation_magnitude
                    
                    # Add slight randomness to x position
                    pos_x += np.random.uniform(-corridor_width/10, corridor_width/10)
                    
                    # Ensure position is within world boundaries
                    pos_x = np.clip(pos_x, 0, fbr_world_size)
                    pos_y = np.clip(pos_y, 0, fbr_world_size)
                    
                    pos = np.array([pos_x, pos_y, np.random.uniform(0, fbr_world_size * 0.05)])
                else:
                    # Place remaining nodes randomly throughout the world
                    pos = np.array([
                        np.random.uniform(0, fbr_world_size),
                        np.random.uniform(0, fbr_world_size),
                        np.random.uniform(0, fbr_world_size * 0.05)  # Limit depth
                    ])
                
                # Create the node with appropriate energy and range
                node = Node(i, pos, max_range=fbr_max_range, energy=fbr_energy)
                router.add_node(node)
            
            # Set beam width in radians (conversion from degrees)
            router.beam_width = np.radians(fbr_beam_width)
            
            # Set power levels
            router.power_levels = fbr_power_levels
            
            # Display network visualization
            st.subheader("FBR Network Visualization")
            fig = plt.figure(figsize=(10, 8))
            router.visualize_network_2d()
            show_figure(plt.gcf())
            
            # Route a packet
            st.subheader("FBR Routing Simulation")
            
            # Create a data packet to send
            data = "Hello from source to sink"
            
            # Fix: Patch the find_next_hop method to prevent infinite recursion
            original_find_next_hop = router.find_next_hop
            
            # Define a safer version of find_next_hop that won't cause recursion errors
            def safe_find_next_hop(source_node, destination_node, recursion_depth=0):
                """Safe implementation to find next hop without infinite recursion"""
                if recursion_depth > 2:  # Limit recursion depth
                    return None
                    
                if source_node.node_id == destination_node.node_id:
                    return None  # Already at destination
                    
                # Direct beam toward destination
                beam_direction = source_node.steer_beam(destination_node)
                
                # Find nodes within the beam
                nodes_in_beam = []
                for neighbor_id, neighbor in source_node.neighbors.items():
                    if source_node.is_in_beam(neighbor, beam_direction):
                        nodes_in_beam.append(neighbor)
                
                if not nodes_in_beam:
                    # No neighbors in beam, try to adapt the beam width
                    source_node.beam_width = min(90, source_node.beam_width + 15)
                    
                    # Try again with the wider beam
                    nodes_in_beam = []
                    for neighbor_id, neighbor in source_node.neighbors.items():
                        if source_node.is_in_beam(neighbor, beam_direction):
                            nodes_in_beam.append(neighbor)
                            
                    # If still no nodes in beam, return None
                    if not nodes_in_beam:
                        return None
                
                # Find node closest to destination among those in beam
                next_hop = min(nodes_in_beam, 
                              key=lambda node: node.distance_to(destination_node))
                
                # Check if we're making progress
                if next_hop.distance_to(destination_node) >= source_node.distance_to(destination_node):
                    # We're not getting closer, try beam adaptation
                    nodes_in_vicinity = list(source_node.neighbors.values())
                    source_node.adapt_beam(nodes_in_vicinity, destination_node)
                    
                    # Try one more time with the adapted beam
                    nodes_in_beam = []
                    for neighbor_id, neighbor in source_node.neighbors.items():
                        if source_node.is_in_beam(neighbor, beam_direction):
                            nodes_in_beam.append(neighbor)
                            
                    if nodes_in_beam:
                        next_hop = min(nodes_in_beam, 
                                     key=lambda node: node.distance_to(destination_node))
                        
                        # Check if we're making progress now
                        if next_hop.distance_to(destination_node) < source_node.distance_to(destination_node):
                            return next_hop
                
                    # If we still aren't making progress, return None
                    return None
                    
                return next_hop
            
            # Replace the original method with our safe version
            router.find_next_hop = safe_find_next_hop
            
            # Route packet with the patched method
            success, message = router.route_packet(0, 1, data, max_hops=20, visualize=True)
            
            # Restore original method if needed later
            router.find_next_hop = original_find_next_hop
            
            # Check if routing was successful
            route_result = False
            
            if success:
                # Find the packet in the router's packet list
                packet = next((p for p in router.packets if p.source_id == 0 and p.destination_id == 1), None)
                
                if packet:
                    path = packet.hops
                    route_result = (packet, path, len(path)-1)
            
            if route_result:
                packet, path, hops = route_result
                
                # Show the routing path visualization
                st.subheader("FBR Routing Path")
                fig = plt.figure(figsize=(12, 10))
                router.visualize_network_2d(highlight_path=path)
                show_figure(plt.gcf())
                
                # Display packet information
                st.subheader("FBR Packet Information")
                packet_col1, packet_col2 = st.columns(2)
                
                with packet_col1:
                    st.metric("Packet ID", packet.packet_id)
                    st.metric("Source ID", packet.source_id)
                    st.metric("Destination ID", packet.destination_id)
                
                with packet_col2:
                    st.metric("Hop Count", len(packet.hops) - 1)
                    st.metric("Delivery Status", "Delivered" if packet.delivery_time else "Failed")
                    st.metric("Latency", f"{packet.get_latency():.4f} s")
                
                # Create animation - only if we have sufficient frames
                if router.animation_frames:
                    st.subheader("FBR Packet Animation")
                    
                    # Instead of saving frames to disk, generate them in memory
                    animation_figures = []
                    
                    # Create a proper visualization of the packet movement using animation frames
                    for frame_idx, frame in enumerate(router.animation_frames):
                        fig = plt.figure(figsize=(12, 10))
                        ax = fig.add_subplot(111)
                        
                        # Get the current frame data
                        frame_packet = frame['packet']
                        hop_count = len(frame_packet['hops']) - 1
                        
                        # Draw all nodes as they appear in this frame
                        for node_id, node_data in frame['nodes'].items():
                            pos = node_data['position']
                            state = node_data['state']
                            
                            # Determine node appearance
                            if node_id in path:
                                if node_id == path[0]:
                                    color = 'green'  # Source node
                                    size = 150
                                elif node_id == path[-1]:
                                    color = 'red'    # Destination node
                                    size = 150
                                else:
                                    color = 'orange'  # Path node
                                    size = 100
                            else:
                                color = 'blue'  # Regular node
                                size = 50
                            
                            # Add node state indicators
                            edgecolor = 'black'
                            linewidth = 1
                            
                            # Draw the node
                            ax.scatter(pos[0], pos[1], color=color, s=size, 
                                      edgecolor=edgecolor, linewidth=linewidth, zorder=10)
                            
                            # Add node label with ID
                            ax.text(pos[0] + 30, pos[1] + 30, f'Node {node_id}', 
                                   size=8, zorder=10)
                            
                            # Add range circle
                            range_circle = plt.Circle(pos, fbr_max_range, fill=False, 
                                                   alpha=0.1, linestyle='--', edgecolor='gray')
                            ax.add_patch(range_circle)
                        
                        # Draw connections for visited path
                        if len(frame_packet['hops']) > 1:
                            for i in range(len(frame_packet['hops']) - 1):
                                start_id = frame_packet['hops'][i]
                                end_id = frame_packet['hops'][i + 1]
                                
                                start_pos = frame['nodes'][start_id]['position']
                                end_pos = frame['nodes'][end_id]['position']
                                
                                # Draw connection line
                                ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                                       'r-', linewidth=2, zorder=5)
                        
                        # Draw the packet at its current position
                        if 'position' in frame_packet:
                            packet_pos = frame_packet['position']
                            ax.scatter(packet_pos[0], packet_pos[1], color='yellow', s=120, 
                                      edgecolor='black', marker='*', zorder=15)
                        
                        # Set title and labels
                        if hop_count == 0:
                            ax.set_title("FBR: Packet Generated at Source", fontsize=16, fontweight='bold')
                        else:
                            ax.set_title(f"FBR: Packet at Hop #{hop_count}", fontsize=16, fontweight='bold')
                        
                        ax.set_xlabel('X Position (m)', fontsize=12)
                        ax.set_ylabel('Y Position (m)', fontsize=12)
                        
                        # Add progress indicator
                        frame_progress = frame_idx / len(router.animation_frames)
                        plt.figtext(0.5, 0.01, f"Progress: {frame_progress*100:.1f}%", 
                                   ha='center', fontsize=12, 
                                   bbox=dict(facecolor='white', alpha=0.8))
                        
                        # Set equal aspect and grid
                        ax.set_aspect('equal')
                        ax.grid(True, alpha=0.3)
                        
                        # Set limits
                        ax.set_xlim(0, fbr_world_size)
                        ax.set_ylim(0, fbr_world_size)
                        
                        # Add figure to our collection (instead of saving to disk)
                        animation_figures.append(fig)
                    
                    try:
                        # Create and display the animation if we have figures
                        if animation_figures:
                            # Generate animation in memory
                            animation_data = create_animation_from_figures(animation_figures, fps=1)
                            # Display the animation
                            st.image(animation_data)
                        else:
                            st.warning("No animation frames were generated. Using static visualization instead.")
                            # Show static routing path
                            fig = plt.figure(figsize=(12, 10))
                            router.visualize_network_2d(highlight_path=path)
                            show_figure(plt.gcf())
                    except Exception as e:
                        st.error(f"Error creating animation: {str(e)}. Showing individual frames instead.")
                        # Show first and last frame if available
                        if animation_figures and len(animation_figures) > 0:
                            # Show first frame
                            show_figure(animation_figures[0])
                            # Show last frame if there's more than one
                            if len(animation_figures) > 1:
                                show_figure(animation_figures[-1])
                
                # Plot performance metrics
                st.subheader("FBR Performance Analysis")
                
                # Use the router's built-in metrics visualization
                fig = router.plot_performance_metrics()
                show_figure(fig)
                
                # Display key metrics in a more readable format
                st.subheader("Key Performance Metrics")
                
                # Calculate summary metrics
                energy_values = list(router.performance_metrics['energy_consumption'].values())
                avg_energy = np.mean(energy_values) if energy_values else 0
                
                delays = router.performance_metrics['transmission_delays']
                avg_delay = np.mean(delays) if delays else 0
                
                hop_counts = router.performance_metrics['hop_counts']
                avg_hops = np.mean(hop_counts) if hop_counts else 0
                
                # Display metrics in columns
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric("Total Nodes", len(router.nodes))
                    st.metric("Packet Hop Count", len(path) - 1)
                
                with metric_col2:
                    st.metric("Average Energy Used", f"{avg_energy:.2f}")
                    st.metric("End-to-End Delay", f"{packet.get_latency():.4f} s")
                
                with metric_col3:
                    st.metric("Beam Width", f"{fbr_beam_width}°")
                    st.metric("Transmission Range", f"{fbr_max_range} m")
                
                # Show beam width effects
                st.subheader("Beam Width Effects Analysis")
                
                # Run tests with different beam widths
                beam_test_results = []
                
                for test_width in [15, 30, 45, 60, 75, 90]:
                    # Test this beam width
                    router.beam_width = np.radians(test_width)
                    test_success, _ = router.route_packet(0, 1, f"Test with {test_width}°", visualize=False)
                    
                    # Record result
                    if test_success:
                        test_packet = router.packets[-1]
                        beam_test_results.append({
                            'width': test_width,
                            'success': True,
                            'hops': len(test_packet.hops) - 1,
                            'latency': test_packet.get_latency()
                        })
                    else:
                        beam_test_results.append({
                            'width': test_width,
                            'success': False,
                            'hops': 0,
                            'latency': 0
                        })
                
                # Create beam width analysis visualization
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                # Extract data
                widths = [r['width'] for r in beam_test_results]
                success = [1 if r['success'] else 0 for r in beam_test_results]
                hops = [r['hops'] for r in beam_test_results]
                
                # Plot success rate by beam width
                ax1.bar(widths, success, color='green', alpha=0.7)
                ax1.set_xlabel('Beam Width (degrees)')
                ax1.set_ylabel('Success (1 = Yes, 0 = No)')
                ax1.set_title('Routing Success by Beam Width')
                ax1.set_xticks(widths)
                ax1.grid(axis='y', linestyle='--', alpha=0.6)
                
                # Plot hop count by beam width (only for successful routes)
                successful_widths = [r['width'] for r in beam_test_results if r['success']]
                successful_hops = [r['hops'] for r in beam_test_results if r['success']]
                
                if successful_widths:  # Only plot if we have successful routes
                    ax2.plot(successful_widths, successful_hops, 'bo-', linewidth=2)
                    ax2.set_xlabel('Beam Width (degrees)')
                    ax2.set_ylabel('Hop Count')
                    ax2.set_title('Hop Count by Beam Width')
                    ax2.grid(True, alpha=0.6)
                else:
                    ax2.text(0.5, 0.5, 'No successful routes to analyze', 
                            ha='center', va='center', fontsize=12)
                
                plt.tight_layout()
                show_figure(fig)
                
            else:
                st.error(f"Routing failed: {message}")
                st.warning("Try adjusting the parameters. Consider:")
                st.markdown("""
                - Increasing the node count for better connectivity
                - Increasing the beam width for wider coverage
                - Increasing the max range for longer transmission distance
                - Adjusting the power levels for more transmission options
                """)
                
                # Show the network visualization
                st.subheader("Current Network Topology")
                fig = plt.figure(figsize=(10, 8))
                router.visualize_network_2d()
                show_figure(plt.gcf())
                
                # Show node positions and connectivity
                st.subheader("Node Connectivity Analysis")
                
                # Calculate and display network statistics
                total_connections = sum(len(node.neighbors) for node in router.nodes.values())
                avg_connections = total_connections / len(router.nodes)
                
                # Display connectivity metrics
                conn_col1, conn_col2 = st.columns(2)
                
                with conn_col1:
                    st.metric("Total Nodes", len(router.nodes))
                    st.metric("Average Connections", f"{avg_connections:.2f}")
                
                with conn_col2:
                    st.metric("Max Range", f"{fbr_max_range} m")
                    st.metric("Beam Width", f"{fbr_beam_width}°")

#--------------------- UWSN Clustering Tab ---------------------#
with tab3:
    st.header("UWSN Clustering Protocol")
    st.markdown("""
    The UWSN Clustering Protocol organizes nodes into clusters, with cluster heads aggregating data from cluster members
    and transmitting it to the sink. This approach reduces energy consumption and improves network lifetime.
    """)
    
    # Input parameters for UWSN Clustering
    st.subheader("Simulation Parameters")
    uwsn_col1, uwsn_col2, uwsn_col3 = st.columns(3)
    
    with uwsn_col1:
        uwsn_num_nodes = st.slider("Number of Nodes", 20, 200, 50, step=5, key="uwsn_num_nodes")
        uwsn_world_size = st.slider("World Size", 500, 2000, 1000, step=100, key="uwsn_world_size")
    
    with uwsn_col2:
        uwsn_init_energy = st.slider("Initial Energy", 50, 200, 100, step=10, key="uwsn_init_energy")
        uwsn_transmission_range = st.slider("Transmission Range", 50, 500, 200, step=10, key="uwsn_trans_range")
    
    with uwsn_col3:
        uwsn_num_ch = st.slider("Number of Cluster Heads", 1, 10, 3, step=1, key="uwsn_num_ch")
        uwsn_num_frames = st.slider("Number of Simulation Frames", 5, 20, 10, step=1, key="uwsn_num_frames")
    
    # Button to run UWSN Clustering simulation
    if st.button("Run UWSN Clustering Simulation", key="run_uwsn"):
        with st.spinner("Running UWSN Clustering simulation..."):
            # Create a custom UWSN Clustering protocol class for UI usage
            class CustomUWSNClusteringProtocol(UWSNClusteringProtocol):
                def __init__(self, n_nodes, world_size, init_energy, num_cluster_heads, transmission_range=None):
                    super().__init__(n_nodes, world_size, init_energy, num_cluster_heads)
                    if transmission_range is not None:
                        self.transmission_range = transmission_range
            
            # Create the simulation
            uwsn_sim = CustomUWSNClusteringProtocol(
                n_nodes=uwsn_num_nodes,
                world_size=uwsn_world_size,
                init_energy=uwsn_init_energy,
                num_cluster_heads=uwsn_num_ch,
                transmission_range=uwsn_transmission_range
            )
            
            # Select cluster heads and form clusters
            uwsn_sim.select_cluster_heads()
            uwsn_sim.form_clusters()
            
            # Create a figure for the visualization
            st.subheader("UWSN Clustering Network Visualization")
            
            # Create figure for static visualization
            fig, ax = plt.subplots(figsize=(10, 8))
            plt.cla()
            
            # Plot regular nodes
            regular_nodes = np.array([i for i in range(uwsn_sim.n_nodes) if i not in uwsn_sim.cluster_heads])
            if len(regular_nodes) > 0:
                colors = ['gray' if uwsn_sim.energy_levels[node] <= 0 else 'blue' for node in regular_nodes]
                plt.scatter(uwsn_sim.positions[regular_nodes, 0],
                           uwsn_sim.positions[regular_nodes, 1],
                           c=colors, label='Sensor Nodes', alpha=0.6)
            
            # Plot cluster heads
            plt.scatter(uwsn_sim.positions[uwsn_sim.cluster_heads, 0],
                       uwsn_sim.positions[uwsn_sim.cluster_heads, 1],
                       c='red', marker='^', s=100, label='Cluster Heads')
            
            # Plot sink node
            plt.scatter(uwsn_sim.sink_position[0], uwsn_sim.sink_position[1], 
                       c='purple', marker='s', s=150, label='Sink')
            
            # Draw cluster boundaries and connections
            for ch in uwsn_sim.cluster_heads:
                # Draw transmission range circles for cluster heads
                circle = plt.Circle((uwsn_sim.positions[ch, 0], uwsn_sim.positions[ch, 1]), 
                                  uwsn_sim.transmission_range, 
                                  fill=False, 
                                  linestyle='--', 
                                  color='gray', 
                                  alpha=0.3)
                plt.gca().add_patch(circle)
                
                # Draw lines between cluster heads and their member nodes
                for member in uwsn_sim.clusters[ch]:
                    if uwsn_sim.energy_levels[member] > 0:  # Only draw lines for active nodes
                        plt.plot([uwsn_sim.positions[ch, 0], uwsn_sim.positions[member, 0]],
                                [uwsn_sim.positions[ch, 1], uwsn_sim.positions[member, 1]],
                                'g-', alpha=0.2)
            
            plt.xlabel('X Distance (m)')
            plt.ylabel('Y Distance (m)')
            plt.title('UWSN Adaptive Clustering Protocol Simulation')
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            plt.xlim(0, uwsn_sim.world_size)
            plt.ylim(0, uwsn_sim.world_size)
            plt.legend()
            
            # Show the static visualization
            show_figure(fig)
            
            # Display network metrics
            st.subheader("UWSN Clustering Network Metrics")
            
            uwsn_metric_col1, uwsn_metric_col2, uwsn_metric_col3 = st.columns(3)
            
            with uwsn_metric_col1:
                st.metric("Number of Nodes", uwsn_sim.n_nodes)
                st.metric("Number of Cluster Heads", len(uwsn_sim.cluster_heads))
            
            with uwsn_metric_col2:
                total_members = sum(len(members) for members in uwsn_sim.clusters.values())
                st.metric("Total Cluster Members", total_members)
                st.metric("Active Nodes", f"{np.sum(uwsn_sim.energy_levels > 0)}/{uwsn_sim.n_nodes}")
            
            with uwsn_metric_col3:
                avg_members_per_ch = total_members / max(1, len(uwsn_sim.cluster_heads))
                st.metric("Avg. Members per CH", f"{avg_members_per_ch:.1f}")
                st.metric("Network Coverage", f"{100 * np.sum(uwsn_sim.energy_levels > 0) / uwsn_sim.n_nodes:.1f}%")
            
            # Generate animation frames
            st.subheader("UWSN Cluster Formation Animation")
            
            # Create animation frames in memory
            animation_figures = []
            
            # Run for 10 frames (complete clustering process)
            num_frames = 10
            for frame in range(num_frames):
                # Create a figure for this frame
                fig = plt.figure(figsize=(10, 8))
                
                # Visualize network in this frame
                uwsn_sim.visualize_network(frame)
                
                # Don't display each frame, just add to our collection
                animation_figures.append(fig)
                plt.close(fig)
            
            # Create animation from figures
            try:
                if animation_figures:
                    # Generate in-memory animation
                    animation_data = create_animation_from_figures(animation_figures, fps=1)
                    
                    # Display animation
                    st.image(animation_data)
                else:
                    st.warning("No animation frames were generated.")
                    # Display a static visualization as fallback
                    fig = plt.figure(figsize=(10, 8))
                    uwsn_sim.visualize_network(9)  # Last frame
                    show_figure(fig)
            except Exception as e:
                st.error(f"Animation error: {str(e)}. Showing static visualization instead.")
                # Show just the final state
                fig = plt.figure(figsize=(10, 8)) 
                uwsn_sim.visualize_network(9)  # Show final frame
                show_figure(fig)
                
            # Run multiple rounds to gather performance metrics
            for _ in range(5):
                uwsn_sim.simulate_data_transfer()
            
            # Plot performance metrics
            st.subheader("UWSN Clustering Performance Analysis")
            
            # Create energy consumption and delay plot
            fig, axs = plt.subplots(1, 3, figsize=(18, 5))
            
            # Plot energy consumption
            axs[0].plot(uwsn_sim.energy_consumption, label='Energy Consumption', color='blue')
            axs[0].set_xlabel('Time Frame')
            axs[0].set_ylabel('Energy (Joules)')
            axs[0].set_title('Energy Consumption Over Time')
            axs[0].grid(True)
            
            # Plot transmission delay
            axs[1].plot(uwsn_sim.transmission_delay, label='Transmission Delay', color='red')
            axs[1].set_xlabel('Time Frame')
            axs[1].set_ylabel('Delay (ms)')
            axs[1].set_title('Transmission Delay Over Time')
            axs[1].grid(True)
            
            # Plot node status
            axs[2].plot(uwsn_sim.active_nodes, label='Active Nodes', color='green')
            axs[2].plot(uwsn_sim.dead_nodes, label='Dead Nodes', color='gray')
            axs[2].set_xlabel('Time Frame')
            axs[2].set_ylabel('Number of Nodes')
            axs[2].set_title('Active and Dead Nodes Over Time')
            axs[2].legend()
            axs[2].grid(True)
            
            plt.tight_layout()
            
            # Show the performance metrics visualization
            show_figure(fig)
            
            # Calculate and display average metrics
            avg_energy_consumption = np.mean(uwsn_sim.energy_consumption) if uwsn_sim.energy_consumption else 0
            avg_delay = np.mean(uwsn_sim.transmission_delay) if uwsn_sim.transmission_delay else 0
            
            st.subheader("UWSN Clustering Average Performance Metrics")
            
            avg_metric_col1, avg_metric_col2 = st.columns(2)
            
            with avg_metric_col1:
                st.metric("Average Energy Consumption", f"{avg_energy_consumption:.6f} J")
            
            with avg_metric_col2:
                st.metric("Average Transmission Delay", f"{avg_delay:.6f} s")

# Add protocol comparison section
st.header("Protocol Comparison")
st.markdown("""
Use the tabs above to simulate each protocol with custom parameters. Once you've run multiple simulations,
you can compare their performance metrics here.

Key comparison points:
- **Energy Efficiency**: Lower energy consumption means longer network lifetime
- **Packet Delivery Ratio**: Higher PDR means more reliable communication
- **End-to-End Delay**: Lower delay means faster data delivery
- **Scalability**: How well the protocol handles increasing numbers of nodes
""")

st.markdown("""
### Protocol Selection Guidelines

**Vector-Based Forwarding (VBF)**
- Best for: Sparse networks, mobile nodes, scenarios requiring controlled forwarding paths
- Advantages: Reduces unnecessary transmissions, handles node mobility well
- Challenges: Performance depends on pipe radius parameter, may have higher delays

**Focused Beam Routing (FBR)**
- Best for: Energy-constrained networks, 3D underwater deployments
- Advantages: Efficient energy usage, adapts to network conditions
- Challenges: Requires precise location information, can be complex to implement

**UWSN Clustering**
- Best for: Dense networks, data aggregation scenarios
- Advantages: Reduces network traffic, extends overall network lifetime
- Challenges: Cluster head selection is critical, potential single points of failure
""")

# Footer
st.markdown("---")
st.markdown("**UWSN Protocol Simulator** - A tool for simulating and comparing underwater sensor network routing protocols") 