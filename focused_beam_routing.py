#!/usr/bin/env python3

import math
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Wedge, Circle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.proj3d import proj_transform
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import time
import random

# Arrow in 3D for matplotlib
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

class NodeState(Enum):
    IDLE = 0
    TRANSMITTING = 1
    RECEIVING = 2
    FORWARDING = 3

class Packet:
    def __init__(self, packet_id, source_id, destination_id, data, size=64):
        self.packet_id = packet_id
        self.source_id = source_id
        self.destination_id = destination_id
        self.data = data
        self.size = size  # Size in bytes
        self.creation_time = time.time()
        self.delivery_time = None
        self.hops = []
        
    def add_hop(self, node_id):
        self.hops.append(node_id)
    
    def mark_delivered(self):
        self.delivery_time = time.time()
    
    def get_latency(self):
        if self.delivery_time:
            return self.delivery_time - self.creation_time
        return None
    
    def __str__(self):
        return f"Packet {self.packet_id}: {self.source_id} → {self.destination_id} ({self.size} bytes)"

class Node:
    def __init__(self, node_id, position, max_range=1000, energy=100):
        self.node_id = node_id
        self.position = position  # (x, y, z) coordinates in meters
        self.max_range = max_range  # maximum transmission range in meters
        self.neighbors = {}  # dictionary of neighbor nodes {node_id: Node}
        self.state = NodeState.IDLE
        self.beam_width = 30  # beam width in degrees
        self.current_beam_direction = None  # (azimuth, elevation) in degrees
        self.energy = energy  # Battery level (0-100%)
        self.packets_sent = 0
        self.packets_received = 0
        self.packets_forwarded = 0
        self.sensor_data = {}  # Simulated sensor readings
        self.color = 'blue'
        
    def generate_sensor_data(self):
        """Generate random sensor data"""
        self.sensor_data = {
            'temperature': round(random.uniform(20, 30), 1),
            'humidity': round(random.uniform(30, 90), 1),
            'pressure': round(random.uniform(980, 1020), 1),
            'battery': self.energy
        }
        return self.sensor_data
        
    def transmit_packet(self, packet):
        """Transmit a packet - consumes energy"""
        self.state = NodeState.TRANSMITTING
        self.packets_sent += 1
        # Energy consumption model (simplistic)
        self.energy -= 0.1 * (self.beam_width / 30)  # Wider beams use more energy
        return True
    
    def receive_packet(self, packet):
        """Receive a packet"""
        self.state = NodeState.RECEIVING
        self.packets_received += 1
        # Energy consumption model (simplistic)
        self.energy -= 0.05
        return True
    
    def forward_packet(self, packet):
        """Forward a packet"""
        self.state = NodeState.FORWARDING
        self.packets_forwarded += 1
        # Energy consumption model (simplistic)
        self.energy -= 0.15
        return True
    
    def distance_to(self, other_node):
        """Calculate Euclidean distance to another node"""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(self.position, other_node.position)))
    
    def angle_to(self, other_node):
        """Calculate azimuth and elevation angles to another node"""
        dx = other_node.position[0] - self.position[0]
        dy = other_node.position[1] - self.position[1]
        dz = other_node.position[2] - self.position[2]
        
        # Azimuth angle (horizontal plane)
        azimuth = math.degrees(math.atan2(dy, dx))
        
        # Elevation angle (vertical plane)
        horizontal_distance = math.sqrt(dx**2 + dy**2)
        elevation = math.degrees(math.atan2(dz, horizontal_distance))
        
        return (azimuth, elevation)
    
    def is_in_beam(self, other_node, beam_direction):
        """Check if another node is within the beam cone"""
        target_angle = self.angle_to(other_node)
        
        # Calculate angular difference
        azimuth_diff = abs((target_angle[0] - beam_direction[0] + 180) % 360 - 180)
        elevation_diff = abs((target_angle[1] - beam_direction[1] + 180) % 360 - 180)
        
        # Return true if within beam width and range
        return (azimuth_diff <= self.beam_width/2 and 
                elevation_diff <= self.beam_width/2 and
                self.distance_to(other_node) <= self.max_range)
    
    def steer_beam(self, target_node):
        """Direct beam toward a specific target node"""
        self.current_beam_direction = self.angle_to(target_node)
        return self.current_beam_direction
    
    def adapt_beam(self, nodes_in_vicinity, target_node):
        """Adaptively adjust beam parameters based on network conditions"""
        # Start with direct beam to target
        ideal_direction = self.angle_to(target_node)
        
        # Distance to target (affects beam width)
        distance_to_target = self.distance_to(target_node)
        distance_factor = min(1.0, distance_to_target / self.max_range)
        
        # Set initial beam width based on node density and distance
        if len(nodes_in_vicinity) > 10:
            # In dense areas, use narrower beam to reduce interference
            self.beam_width = max(15, min(self.beam_width, 30))
        elif len(nodes_in_vicinity) > 5:
            # Medium density - moderate beam width
            self.beam_width = max(25, min(self.beam_width, 60)) 
        else:
            # In sparse areas, use wider beam to increase connectivity
            self.beam_width = max(40, min(90, self.beam_width + 15))
            
        # Increase beam width for long-distance transmissions
        if distance_factor > 0.7:  # Far targets need wider beams
            self.beam_width = min(90, self.beam_width + 15)
        
        # Set the beam direction directly to target
        self.current_beam_direction = ideal_direction
        return self.current_beam_direction


class FocusedBeamRouter:
    def __init__(self):
        self.nodes = {}  # dictionary of all nodes {node_id: Node}
        self.packets = []  # List of packets in the network
        self.current_packet_id = 0
        self.animation_frames = []  # For storing animation frames
        self.performance_metrics = {
            'energy_consumption': {},  # node_id -> energy used
            'transmission_delays': [],  # list of hop delays
            'hop_counts': [],          # list of hop counts
            'beam_widths': []          # list of beam widths used
        }
        self.power_levels = 3  # Default power levels
        self.beam_width = np.radians(30)  # Default beam width in radians
    
    def add_node(self, node):
        """Add a node to the network"""
        self.nodes[node.node_id] = node
        
        # Initialize energy tracking
        self.performance_metrics['energy_consumption'][node.node_id] = 0
        
        # Update neighbor lists
        for node_id, existing_node in self.nodes.items():
            if node_id != node.node_id:
                distance = node.distance_to(existing_node)
                if distance <= node.max_range:
                    node.neighbors[node_id] = existing_node
                if distance <= existing_node.max_range:
                    existing_node.neighbors[node.node_id] = node
    
    def create_packet(self, source_id, destination_id, data):
        """Create a new packet"""
        self.current_packet_id += 1
        packet = Packet(self.current_packet_id, source_id, destination_id, data)
        self.packets.append(packet)
        return packet
    
    def find_next_hop(self, source_node, destination_node):
        """Implement greedy forwarding to find next hop within beam"""
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
            # No neighbors in beam, try to adapt the beam width - but don't recursively call
            source_node.beam_width = min(90, source_node.beam_width + 10)
            
            # Try again with the wider beam, but don't create infinite recursion
            # Find nodes within the wider beam
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
            # We're not getting closer, try beam adaptation without recursion
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
    
    def route_packet(self, source_id, destination_id, data, max_hops=20, visualize=False):
        """Route a packet from source to destination using focused beam routing"""
        if source_id not in self.nodes or destination_id not in self.nodes:
            return False, "Source or destination node not found"
        
        # Record initial energy levels
        initial_energy = {node_id: node.energy for node_id, node in self.nodes.items()}
        
        packet = self.create_packet(source_id, destination_id, data)
        current_node = self.nodes[source_id]
        destination_node = self.nodes[destination_id]
        
        # Add first hop
        packet.add_hop(current_node.node_id)
        
        hops = 0
        hop_timestamps = [time.time()]  # Record time at each hop
        packet_position = None  # For animation
        beam_widths_used = []  # Track beam widths used
        
        # Reset animation frames if visualizing
        if visualize:
            self.animation_frames = []
        
        while current_node.node_id != destination_id and hops < max_hops:
            # Find next hop
            next_node = self.find_next_hop(current_node, destination_node)
            
            # Record beam width used
            beam_widths_used.append(current_node.beam_width)
            
            if next_node is None:
                return False, f"No route found after {hops} hops. Path: {packet.hops}"
            
            # Perform packet forwarding
            current_node.state = NodeState.TRANSMITTING
            next_node.state = NodeState.RECEIVING
            
            # For visualization, show packet moving between nodes
            if visualize:
                # Capture 5 intermediate frames showing packet in transit
                for i in range(5):
                    alpha = i / 4.0  # 0.0, 0.25, 0.5, 0.75, 1.0
                    packet_position = (
                        current_node.position[0] + alpha * (next_node.position[0] - current_node.position[0]),
                        current_node.position[1] + alpha * (next_node.position[1] - current_node.position[1])
                    )
                    frame = self.capture_visualization_frame(packet, current_node, next_node, packet_position)
                    self.animation_frames.append(frame)
            
            # Update packet and node stats
            current_node.transmit_packet(packet)
            next_node.receive_packet(packet)
            
            # Update current node and path
            current_node = next_node
            packet.add_hop(current_node.node_id)
            hops += 1
            
            # Record timestamp for this hop
            hop_timestamps.append(time.time())
            
            # Reset node states
            for node in self.nodes.values():
                node.state = NodeState.IDLE
        
        # Calculate energy consumed by each node
        for node_id, node in self.nodes.items():
            energy_consumed = initial_energy[node_id] - node.energy
            self.performance_metrics['energy_consumption'][node_id] += energy_consumed
        
        # Calculate hop delays
        hop_delays = [hop_timestamps[i+1] - hop_timestamps[i] for i in range(len(hop_timestamps)-1)]
        self.performance_metrics['transmission_delays'].extend(hop_delays)
        
        # Record hop count
        self.performance_metrics['hop_counts'].append(hops)
        
        # Record beam widths
        self.performance_metrics['beam_widths'].extend(beam_widths_used)
        
        if current_node.node_id == destination_id:
            packet.mark_delivered()
            return True, f"Packet delivered in {hops} hops. Path: {packet.hops}"
        else:
            return False, f"Max hops reached. Path: {packet.hops}"
    
    def capture_visualization_frame(self, packet, source_node, target_node, packet_position):
        """Capture a frame for animation"""
        # Create a deep copy of the network state for this frame
        frame = {
            'nodes': {node_id: {
                'position': node.position[:2],  # Only x,y for 2D
                'state': node.state,
                'energy': node.energy,
                'beam_direction': node.current_beam_direction
            } for node_id, node in self.nodes.items()},
            'packet': {
                'id': packet.packet_id,
                'position': packet_position,
                'source': packet.source_id,
                'destination': packet.destination_id,
                'hops': packet.hops.copy()
            },
            'active_source': source_node.node_id,
            'active_target': target_node.node_id
        }
        return frame
    
    def visualize_network_2d(self, highlight_path=None, show_packet=None):
        """Visualize the network in 2D with matplotlib (top-down view)"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Extract node positions (only x and y coordinates)
        node_positions = {node_id: (node.position[0], node.position[1]) 
                         for node_id, node in self.nodes.items()}
        
        # Plot nodes with different colors based on path
        for node_id, pos in node_positions.items():
            node = self.nodes[node_id]
            
            if highlight_path and node_id in highlight_path:
                if node_id == highlight_path[0]:
                    color = 'green'  # Source
                    size = 150
                elif node_id == highlight_path[-1]:
                    color = 'red'    # Destination
                    size = 150
                else:
                    color = 'orange' # Path node
                    size = 100
            else:
                color = 'blue'  # Regular node
                size = 50
                
            # Mark node state
            if node.state == NodeState.TRANSMITTING:
                edgecolor = 'red'
                linewidth = 2
            elif node.state == NodeState.RECEIVING:
                edgecolor = 'green'
                linewidth = 2
            else:
                edgecolor = 'black'
                linewidth = 1
            
            # Plot the node
            ax.scatter(pos[0], pos[1], color=color, s=size, 
                       edgecolor=edgecolor, linewidth=linewidth, zorder=10)
            
            # Add node label with ID and energy level
            ax.text(pos[0] + 50, pos[1] + 50, 
                    f'Node {node_id}\n{node.energy:.1f}%', 
                    size=8, zorder=10)
            
            # Add range indicator (faded circle)
            range_circle = Circle(pos, node.max_range, fill=False, 
                                 alpha=0.1, linestyle='--', edgecolor='gray')
            ax.add_patch(range_circle)
        
        # Draw connections between nodes in the path
        if highlight_path and len(highlight_path) > 1:
            for i in range(len(highlight_path) - 1):
                start_id = highlight_path[i]
                end_id = highlight_path[i + 1]
                start_pos = node_positions[start_id]
                end_pos = node_positions[end_id]
                
                # Draw arrow from start to end
                ax.annotate("", 
                           xy=end_pos, xycoords='data',
                           xytext=start_pos, textcoords='data',
                           arrowprops=dict(arrowstyle="->", lw=2, color="red"),
                           zorder=5)
                
                # Visualize beam in 2D
                self._visualize_beam_2d(ax, self.nodes[start_id], self.nodes[end_id])
        
        # Draw packet if provided
        if show_packet:
            packet_x, packet_y = show_packet
            ax.scatter(packet_x, packet_y, color='yellow', s=100, 
                      edgecolor='black', marker='*', zorder=15)
        
        # Set labels and title
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_title('Focused Beam Routing Network 2D Visualization')
        
        # Make the plot more readable with a grid
        ax.grid(True)
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        
        plt.tight_layout()
        return fig, ax
    
    def _visualize_beam_2d(self, ax, source_node, target_node):
        """Visualize the 2D projection of the beam from source to target"""
        # Get source and target positions (x,y only)
        source_pos = (source_node.position[0], source_node.position[1])
        target_pos = (target_node.position[0], target_node.position[1])
        
        # Calculate direction vector in 2D
        dx = target_pos[0] - source_pos[0]
        dy = target_pos[1] - source_pos[1]
        
        # Calculate distance in 2D
        distance = math.sqrt(dx**2 + dy**2)
        
        # Calculate beam length
        beam_length = min(distance, source_node.max_range)
        
        # Get beam angle in radians
        beam_angle_rad = math.radians(source_node.beam_width)
        
        # Calculate azimuth angle
        azimuth = math.degrees(math.atan2(dy, dx))
        
        # Create a wedge to represent the beam
        wedge = Wedge(
            source_pos,
            beam_length,  # radius
            azimuth - source_node.beam_width/2,  # start angle
            azimuth + source_node.beam_width/2,  # end angle
            width=None,
            alpha=0.3,
            color='darkblue',
            zorder=1
        )
        
        ax.add_patch(wedge)
        
    def animate_packet_routing(self, packet_path):
        """Create an animation of the packet routing process"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        def init():
            ax.clear()
            return []
        
        def update(frame_num):
            ax.clear()
            
            if frame_num >= len(self.animation_frames):
                frame_num = len(self.animation_frames) - 1
                
            frame = self.animation_frames[frame_num]
            
            # Draw all nodes
            for node_id, node_data in frame['nodes'].items():
                pos = node_data['position']
                state = node_data['state']
                energy = node_data['energy']
                
                # Set node appearance based on its role in the path
                if node_id in packet_path:
                    if node_id == packet_path[0]:
                        color = 'green'  # Source
                        size = 150
                    elif node_id == packet_path[-1]:
                        color = 'red'    # Destination
                        size = 150
                    else:
                        color = 'orange'  # Path node
                        size = 100
                else:
                    color = 'blue'  # Regular node
                    size = 50
                    
                # Modify node appearance based on state
                if state == NodeState.TRANSMITTING:
                    edgecolor = 'red'
                    linewidth = 2
                elif state == NodeState.RECEIVING:
                    edgecolor = 'green'
                    linewidth = 2
                else:
                    edgecolor = 'black'
                    linewidth = 1
                
                # Draw the node
                ax.scatter(pos[0], pos[1], color=color, s=size, 
                          edgecolor=edgecolor, linewidth=linewidth, zorder=10)
                
                # Add node label
                ax.text(pos[0] + 50, pos[1] + 50, 
                       f'Node {node_id}\n{energy:.1f}%', 
                       size=8, zorder=10)
                
                # Add range indicator
                range_circle = Circle(pos, 1000, fill=False, 
                                    alpha=0.1, linestyle='--', edgecolor='gray')
                ax.add_patch(range_circle)
            
            # Draw connections along the path
            for i in range(len(packet_path) - 1):
                # Only draw connections that have been traversed so far
                if packet_path[i+1] in frame['packet']['hops']:
                    start_id = packet_path[i]
                    end_id = packet_path[i+1]
                    
                    start_pos = frame['nodes'][start_id]['position']
                    end_pos = frame['nodes'][end_id]['position']
                    
                    # Draw arrow from start to end
                    ax.annotate("", 
                               xy=end_pos, xycoords='data',
                               xytext=start_pos, textcoords='data',
                               arrowprops=dict(arrowstyle="->", lw=2, color="red"),
                               zorder=5)
            
            # Draw active beam if applicable
            if 'active_source' in frame and 'active_target' in frame:
                source_id = frame['active_source']
                target_id = frame['active_target']
                
                source_pos = frame['nodes'][source_id]['position']
                target_pos = frame['nodes'][target_id]['position']
                
                # Calculate beam parameters
                dx = target_pos[0] - source_pos[0]
                dy = target_pos[1] - source_pos[1]
                distance = math.sqrt(dx**2 + dy**2)
                beam_length = min(distance, 1000)
                beam_width = 30  # Degrees
                azimuth = math.degrees(math.atan2(dy, dx))
                
                # Create a wedge to represent the beam
                wedge = Wedge(
                    source_pos,
                    beam_length,  # radius
                    azimuth - beam_width/2,  # start angle
                    azimuth + beam_width/2,  # end angle
                    width=None,
                    alpha=0.4,
                    color='darkblue',
                    zorder=1
                )
                ax.add_patch(wedge)
            
            # Draw the packet if it exists in this frame
            if 'packet' in frame and 'position' in frame['packet']:
                packet_pos = frame['packet']['position']
                ax.scatter(packet_pos[0], packet_pos[1], color='yellow', 
                          s=100, edgecolor='black', marker='*', zorder=15)
                
                # Add packet info text
                packet_id = frame['packet']['id']
                source_id = frame['packet']['source']
                dest_id = frame['packet']['destination']
                ax.text(packet_pos[0] + 50, packet_pos[1] - 50, 
                       f'Packet {packet_id}\n{source_id} → {dest_id}', 
                       size=8, zorder=15)
            
            # Set labels and title
            ax.set_xlabel('X (meters)')
            ax.set_ylabel('Y (meters)')
            hop_count = len(frame['packet']['hops'])
            ax.set_title(f'Focused Beam Routing: Packet Transfer (Hop {hop_count-1})')
            
            # Additional info text
            info_text = f"Packet ID: {frame['packet']['id']}\n"
            info_text += f"Source: Node {frame['packet']['source']}\n"
            info_text += f"Destination: Node {frame['packet']['destination']}\n"
            info_text += f"Current Hops: {frame['packet']['hops']}"
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=9, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # Set grid and aspect ratio
            ax.grid(True)
            ax.set_aspect('equal')
            
            return []
        
        # Create the animation
        ani = animation.FuncAnimation(fig, update, frames=len(self.animation_frames),
                                     init_func=init, blit=True, interval=200)
        
        return ani

    def plot_performance_metrics(self):
        """Plot performance metrics for the routing protocol"""
        # Use a more modern style for plots
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Create a figure with 2x2 subplots
        fig, axs = plt.subplots(2, 2, figsize=(16, 12), dpi=100)
        fig.suptitle('Focused Beam Routing Protocol Performance Metrics', fontsize=18, fontweight='bold', y=0.98)
        
        # Color palette
        colors = {
            'energy': '#FF6B6B',       # Red-ish
            'delay': '#4ECDC4',        # Teal
            'hops': '#9D79BC',         # Purple
            'beam': '#4056A1',         # Blue
            'mean_line': '#F13C20',    # Bright red
            'median_line': '#2E5EAA',  # Dark blue
            'bar_edge': '#2F2D2E',     # Dark gray
            'grid': '#CCCCCC'          # Light gray
        }
        
        # Common styling
        title_properties = {
            'fontsize': 14,
            'fontweight': 'bold',
            'pad': 15
        }
        
        label_properties = {
            'fontsize': 11,
            'fontweight': 'medium',
            'labelpad': 10
        }
        
        # Plot 1: Energy consumption per node
        node_ids = list(self.performance_metrics['energy_consumption'].keys())
        energy_values = list(self.performance_metrics['energy_consumption'].values())
        
        # Sort by energy consumption
        sorted_indices = sorted(range(len(energy_values)), key=lambda i: energy_values[i], reverse=True)
        sorted_node_ids = [node_ids[i] for i in sorted_indices]
        sorted_energy_values = [energy_values[i] for i in sorted_indices]
        
        # Take top 15 energy consumers for readability
        display_nodes = min(15, len(sorted_node_ids))
        
        # Plot bars with nice styling
        bars = axs[0, 0].bar(
            range(display_nodes), 
            sorted_energy_values[:display_nodes], 
            color=colors['energy'],
            edgecolor=colors['bar_edge'],
            alpha=0.8,
            linewidth=1,
            width=0.7
        )
        
        # Add values on top of bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0.05:  # Only add text if bar is tall enough
                axs[0, 0].text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 0.01,
                    f"{sorted_energy_values[i]:.2f}",
                    ha='center',
                    fontsize=9,
                    fontweight='bold'
                )
        
        # Styling
        axs[0, 0].set_xticks(range(display_nodes))
        axs[0, 0].set_xticklabels([f"Node {sorted_node_ids[i]}" for i in range(display_nodes)], 
                                 rotation=45, ha='right')
        axs[0, 0].set_ylabel('Energy Consumed', **label_properties)
        axs[0, 0].set_title('Energy Consumption by Node (Top 15)', **title_properties)
        axs[0, 0].grid(axis='y', linestyle='--', alpha=0.7, color=colors['grid'])
        axs[0, 0].spines['top'].set_visible(False)
        axs[0, 0].spines['right'].set_visible(False)
        
        # Add average energy consumption line
        if sorted_energy_values:
            mean_energy = np.mean(sorted_energy_values)
            axs[0, 0].axhline(
                mean_energy, 
                color=colors['mean_line'], 
                linestyle='--', 
                linewidth=2,
                label=f'Avg: {mean_energy:.2f}'
            )
            axs[0, 0].legend(loc='upper right', frameon=True, fontsize=10)
        
        # Plot 2: Distribution of transmission delays
        if self.performance_metrics['transmission_delays']:
            delays = self.performance_metrics['transmission_delays']
            
            # Calculate statistics for annotations
            mean_delay = np.mean(delays)
            median_delay = np.median(delays)
            min_delay = np.min(delays)
            max_delay = np.max(delays)
            
            # Determine optimal bin count based on data
            num_bins = min(20, max(5, int(np.sqrt(len(delays)))))
            
            # Plot histogram with nice styling
            axs[0, 1].hist(
                delays, 
                bins=num_bins, 
                color=colors['delay'], 
                alpha=0.7, 
                edgecolor=colors['bar_edge'],
                linewidth=1
            )
            
            # Add mean and median lines
            axs[0, 1].axvline(
                mean_delay, 
                color=colors['mean_line'], 
                linestyle='--', 
                linewidth=2, 
                label=f'Mean: {mean_delay:.4f}s'
            )
            
            axs[0, 1].axvline(
                median_delay, 
                color=colors['median_line'], 
                linestyle=':', 
                linewidth=2, 
                label=f'Median: {median_delay:.4f}s'
            )
            
            # Add statistics textbox
            stats_text = f"Min: {min_delay:.4f}s\nMax: {max_delay:.4f}s\nMean: {mean_delay:.4f}s\nMedian: {median_delay:.4f}s"
            props = dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor=colors['grid'])
            axs[0, 1].text(
                0.95, 0.95, stats_text, 
                transform=axs[0, 1].transAxes, 
                fontsize=10,
                verticalalignment='top', 
                horizontalalignment='right',
                bbox=props
            )
            
            # Styling
            axs[0, 1].set_xlabel('Delay (seconds)', **label_properties)
            axs[0, 1].set_ylabel('Frequency', **label_properties)
            axs[0, 1].set_title('Transmission Delay Distribution', **title_properties)
            axs[0, 1].legend(loc='upper left', frameon=True, fontsize=10)
            axs[0, 1].spines['top'].set_visible(False)
            axs[0, 1].spines['right'].set_visible(False)
        
        # Plot 3: Hop count distribution
        if self.performance_metrics['hop_counts']:
            hop_counts = self.performance_metrics['hop_counts']
            max_hops = max(hop_counts)
            hop_count_freq = {i: hop_counts.count(i) for i in range(1, max_hops + 1)}
            
            # Plot bars with nice styling
            hop_bars = axs[1, 0].bar(
                hop_count_freq.keys(), 
                hop_count_freq.values(), 
                color=colors['hops'],
                edgecolor=colors['bar_edge'],
                alpha=0.8,
                linewidth=1,
                width=0.7
            )
            
            # Add count labels on top of bars
            for bar in hop_bars:
                height = bar.get_height()
                axs[1, 0].text(
                    bar.get_x() + bar.get_width()/2.,
                    height + 0.1,
                    f"{int(height)}",
                    ha='center',
                    fontsize=10,
                    fontweight='bold'
                )
            
            # Add mean line
            mean_hops = np.mean(hop_counts)
            axs[1, 0].axvline(
                mean_hops, 
                color=colors['mean_line'], 
                linestyle='--', 
                linewidth=2, 
                label=f'Mean: {mean_hops:.2f}'
            )
            
            # Add hop efficiency info
            # Calculate the total number of nodes in the network
            total_nodes = len(self.nodes)
            max_possible_hops = total_nodes - 1  # Theoretical maximum
            
            efficiency_text = f"Avg hops: {mean_hops:.2f}\nMax observed: {max_hops}\nEfficiency: {100 * mean_hops / max_possible_hops:.1f}% of max"
            props = dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor=colors['grid'])
            axs[1, 0].text(
                0.95, 0.95, efficiency_text, 
                transform=axs[1, 0].transAxes, 
                fontsize=10,
                verticalalignment='top', 
                horizontalalignment='right',
                bbox=props
            )
            
            # Styling
            axs[1, 0].set_xticks(list(hop_count_freq.keys()))
            axs[1, 0].set_xlabel('Number of Hops', **label_properties)
            axs[1, 0].set_ylabel('Frequency', **label_properties)
            axs[1, 0].set_title('Hop Count Distribution', **title_properties)
            axs[1, 0].legend(loc='upper right', frameon=True, fontsize=10)
            axs[1, 0].spines['top'].set_visible(False)
            axs[1, 0].spines['right'].set_visible(False)
        
        # Plot 4: Beam width distribution
        if self.performance_metrics['beam_widths']:
            beam_widths = self.performance_metrics['beam_widths']
            
            # Calculate statistics
            mean_width = np.mean(beam_widths)
            median_width = np.median(beam_widths)
            min_width = np.min(beam_widths)
            max_width = np.max(beam_widths)
            
            # Determine optimal bin count
            num_bins = min(15, max(5, int(np.sqrt(len(beam_widths)))))
            
            # Plot histogram with nice styling
            axs[1, 1].hist(
                beam_widths, 
                bins=num_bins, 
                color=colors['beam'], 
                alpha=0.7, 
                edgecolor=colors['bar_edge'],
                linewidth=1
            )
            
            # Add mean and median lines
            axs[1, 1].axvline(
                mean_width, 
                color=colors['mean_line'], 
                linestyle='--', 
                linewidth=2, 
                label=f'Mean: {mean_width:.2f}°'
            )
            
            axs[1, 1].axvline(
                median_width, 
                color=colors['median_line'], 
                linestyle=':', 
                linewidth=2, 
                label=f'Median: {median_width:.2f}°'
            )
            
            # Add statistics textbox
            stats_text = f"Min: {min_width:.1f}°\nMax: {max_width:.1f}°\nMean: {mean_width:.2f}°\nMedian: {median_width:.2f}°"
            props = dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor=colors['grid'])
            axs[1, 1].text(
                0.95, 0.95, stats_text, 
                transform=axs[1, 1].transAxes, 
                fontsize=10,
                verticalalignment='top', 
                horizontalalignment='right',
                bbox=props
            )
            
            # Styling
            axs[1, 1].set_xlabel('Beam Width (degrees)', **label_properties)
            axs[1, 1].set_ylabel('Frequency', **label_properties)
            axs[1, 1].set_title('Beam Width Distribution', **title_properties)
            axs[1, 1].legend(loc='upper left', frameon=True, fontsize=10)
            axs[1, 1].spines['top'].set_visible(False)
            axs[1, 1].spines['right'].set_visible(False)
        
        # Add timestamp and simulation info
        simulation_info = f"Simulation: {len(self.nodes)} nodes, {len(self.packets)} packets"
        fig.text(0.5, 0.01, simulation_info, ha='center', fontsize=10, fontstyle='italic')
        
        # Add footer with timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        fig.text(0.99, 0.01, f"Generated: {timestamp}", ha='right', fontsize=8, fontstyle='italic')
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])  # Adjust for the suptitle and footer
        return fig

    def generate_performance_report(self, filename="performance_report.txt", num_nodes=None):
        """Generate a detailed performance report as a text file"""
        with open(filename, 'w') as f:
            f.write("="*60 + "\n")
            f.write("FOCUSED BEAM ROUTING PROTOCOL - PERFORMANCE REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Report generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Energy statistics
            energy_values = list(self.performance_metrics['energy_consumption'].values())
            f.write("1. ENERGY CONSUMPTION:\n")
            f.write(f"   Total energy consumed: {sum(energy_values):.2f}\n")
            f.write(f"   Average energy per node: {np.mean(energy_values):.2f}\n")
            f.write(f"   Max energy consumed by a node: {max(energy_values):.2f}\n")
            f.write(f"   Min energy consumed by a node: {min(energy_values):.2f}\n")
            
            # Top energy consuming nodes
            node_ids = list(self.performance_metrics['energy_consumption'].keys())
            sorted_indices = sorted(range(len(energy_values)), key=lambda i: energy_values[i], reverse=True)
            sorted_node_ids = [node_ids[i] for i in sorted_indices]
            sorted_energy_values = [energy_values[i] for i in sorted_indices]
            
            f.write("\n2. TOP 10 ENERGY CONSUMING NODES:\n")
            for i in range(min(10, len(sorted_node_ids))):
                f.write(f"   Node {sorted_node_ids[i]}: {sorted_energy_values[i]:.2f}\n")
            
            # Transmission delay statistics
            delays = self.performance_metrics['transmission_delays']
            f.write("\n3. TRANSMISSION DELAYS (seconds):\n")
            f.write(f"   Min: {np.min(delays):.6f}\n")
            f.write(f"   Max: {np.max(delays):.6f}\n")
            f.write(f"   Mean: {np.mean(delays):.6f}\n")
            f.write(f"   Median: {np.median(delays):.6f}\n")
            f.write(f"   Standard deviation: {np.std(delays):.6f}\n")
            
            # Hop count statistics
            hop_counts = self.performance_metrics['hop_counts']
            f.write("\n4. HOP COUNTS:\n")
            f.write(f"   Min: {min(hop_counts)}\n")
            f.write(f"   Max: {max(hop_counts)}\n")
            f.write(f"   Mean: {np.mean(hop_counts):.2f}\n")
            f.write(f"   Median: {np.median(hop_counts):.2f}\n")
            f.write(f"   Total hops across all packets: {sum(hop_counts)}\n")
            
            # Create a more readable hop frequency distribution
            hop_count_freq = {}
            for i in range(1, max(hop_counts) + 1):
                count = hop_counts.count(i)
                if count > 0:
                    hop_count_freq[i] = count
            
            f.write("   Hop frequency distribution:\n")
            for hops, count in hop_count_freq.items():
                f.write(f"     {hops} hop(s): {count} packet(s) ({count/len(hop_counts)*100:.1f}%)\n")
            
            # Beam width statistics
            beam_widths = self.performance_metrics['beam_widths']
            f.write("\n5. BEAM WIDTHS (degrees):\n")
            f.write(f"   Min: {np.min(beam_widths):.1f}\n")
            f.write(f"   Max: {np.max(beam_widths):.1f}\n")
            f.write(f"   Mean: {np.mean(beam_widths):.2f}\n")
            f.write(f"   Median: {np.median(beam_widths):.2f}\n")
            f.write(f"   Standard deviation: {np.std(beam_widths):.2f}\n")
            f.write(f"   Total beam adjustments: {len(beam_widths)}\n")
            
            # Network statistics
            if num_nodes is None:
                num_nodes = len(self.nodes)
                
            f.write("\n6. NETWORK STATISTICS:\n")
            f.write(f"   Total nodes: {num_nodes}\n")
            f.write(f"   Total packets routed: {len(self.packets)}\n")
            f.write(f"   Successful deliveries: {len([p for p in self.packets if p.delivery_time])}\n")
            
            # Calculate network density
            total_neighbors = sum(len(node.neighbors) for node in self.nodes.values())
            avg_neighbors = total_neighbors / len(self.nodes)
            f.write(f"   Network density: {avg_neighbors:.2f} neighbors per node\n")
            f.write(f"   Network connectivity: {100 * avg_neighbors / (num_nodes-1):.2f}% of full mesh\n")
            
            # Calculate routing efficiency
            direct_distances = []
            actual_distances = []
            
            for packet in self.packets:
                if packet.delivery_time:  # If successfully delivered
                    source = self.nodes[packet.source_id]
                    dest = self.nodes[packet.destination_id]
                    direct_dist = source.distance_to(dest)
                    direct_distances.append(direct_dist)
                    
                    # Calculate actual path distance
                    actual_dist = 0
                    for i in range(len(packet.hops) - 1):
                        node1 = self.nodes[packet.hops[i]]
                        node2 = self.nodes[packet.hops[i+1]]
                        actual_dist += node1.distance_to(node2)
                    
                    actual_distances.append(actual_dist)
            
            if direct_distances:
                avg_stretch = np.mean([a/d for a, d in zip(actual_distances, direct_distances)])
                f.write(f"   Path stretch factor: {avg_stretch:.2f} (ratio of actual to optimal path length)\n")
            
            f.write("\n7. PACKET DETAILS:\n")
            for i, packet in enumerate(self.packets):
                f.write(f"   Packet {i+1}: Node {packet.source_id} to Node {packet.destination_id}\n")
                if packet.delivery_time:
                    latency = packet.get_latency()
                    hop_count = len(packet.hops) - 1
                    f.write(f"     Status: Delivered in {hop_count} hop(s)\n")
                    f.write(f"     Path: {' -> '.join(map(str, packet.hops))}\n")
                    f.write(f"     Latency: {latency:.6f} seconds\n")
                else:
                    f.write("     Status: Not delivered\n")
                f.write("\n")
            
            f.write("="*60 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*60 + "\n")
            
        return filename

# Example usage
if __name__ == "__main__":
    # Create a sample network
    router = FocusedBeamRouter()
    
    # Parameters for random node placement
    num_nodes = 40  # Reduced to 40 nodes
    area_width = 2000  # meters
    area_height = 2000  # meters
    
    # Create nodes with random positions
    for i in range(1, num_nodes+1):
        # Generate completely random positions
        x = random.uniform(0, area_width)
        y = random.uniform(0, area_height)
        
        # Randomize energy level
        energy = random.uniform(70, 100)
        
        # Add node
        router.add_node(Node(i, (x, y, 0), energy=energy))
    
    # Generate sensor data for each node
    for node in router.nodes.values():
        node.generate_sensor_data()
    
    # Define source and destination for a packet
    # Choose nodes at opposite corners of the area
    nodes_by_position = list(router.nodes.values())
    
    # Find node closest to (0,0) for source
    source_node = min(nodes_by_position, 
                      key=lambda n: math.sqrt(n.position[0]**2 + n.position[1]**2))
    
    # Find node closest to (max,max) for destination
    destination_node = min(nodes_by_position, 
                          key=lambda n: math.sqrt((n.position[0]-area_width)**2 + 
                                                  (n.position[1]-area_height)**2))
    
    source_id = source_node.node_id
    destination_id = destination_node.node_id
    
    print(f"Routing from Node {source_id} to Node {destination_id}")
    
    # Generate and route multiple packets to gather metrics
    num_packets = 5
    for i in range(num_packets):
        # Route packets between different random nodes to collect more diverse metrics
        if i > 0:
            # For subsequent packets, choose random source and destination
            source_id = random.choice(list(router.nodes.keys()))
            # Make sure destination is different from source
            possible_destinations = list(set(router.nodes.keys()) - {source_id})
            destination_id = random.choice(possible_destinations)
            print(f"Routing packet {i+1}: Node {source_id} to Node {destination_id}")
        
        # Create packet data
        data = {
            "type": "sensor_reading", 
            "values": router.nodes[source_id].sensor_data,
            "packet_num": i+1
        }
        
        # Route with visualization only for the first packet
        visualize = (i == 0)
        success, message = router.route_packet(source_id, destination_id, data, visualize=visualize)
        print(message)
    
    # Get the path from the first routing for animation
    first_path_str = router.packets[0].hops
    path = first_path_str
    
    # A small delay to ensure all packets have been processed
    time.sleep(0.5)
    
    # Print performance statistics in terminal
    print("\n" + "="*50)
    print("FOCUSED BEAM ROUTING PROTOCOL - PERFORMANCE STATISTICS")
    print("="*50)
    
    # Energy statistics
    energy_values = list(router.performance_metrics['energy_consumption'].values())
    print(f"\n1. ENERGY CONSUMPTION:")
    print(f"   Total energy consumed: {sum(energy_values):.2f}")
    print(f"   Average energy per node: {np.mean(energy_values):.2f}")
    print(f"   Max energy consumed by a node: {max(energy_values):.2f}")
    
    # Top 5 energy consuming nodes
    node_ids = list(router.performance_metrics['energy_consumption'].keys())
    sorted_indices = sorted(range(len(energy_values)), key=lambda i: energy_values[i], reverse=True)
    sorted_node_ids = [node_ids[i] for i in sorted_indices]
    sorted_energy_values = [energy_values[i] for i in sorted_indices]
    
    print(f"\n2. TOP 5 ENERGY CONSUMING NODES:")
    for i in range(min(5, len(sorted_node_ids))):
        print(f"   Node {sorted_node_ids[i]}: {sorted_energy_values[i]:.2f}")
    
    # Transmission delay statistics
    delays = router.performance_metrics['transmission_delays']
    print(f"\n3. TRANSMISSION DELAYS (seconds):")
    print(f"   Min: {np.min(delays):.4f}")
    print(f"   Max: {np.max(delays):.4f}")
    print(f"   Mean: {np.mean(delays):.4f}")
    print(f"   Median: {np.median(delays):.4f}")
    
    # Hop count statistics
    hop_counts = router.performance_metrics['hop_counts']
    print(f"\n4. HOP COUNTS:")
    print(f"   Min: {min(hop_counts)}")
    print(f"   Max: {max(hop_counts)}")
    print(f"   Mean: {np.mean(hop_counts):.2f}")
    print(f"   Total hops across all packets: {sum(hop_counts)}")
    
    # Create a more readable hop frequency distribution
    hop_count_freq = {}
    for i in range(1, max(hop_counts) + 1):
        count = hop_counts.count(i)
        if count > 0:
            hop_count_freq[i] = count
    
    print("   Hop frequency distribution:")
    for hops, count in hop_count_freq.items():
        print(f"     {hops} hop(s): {count} packet(s)")
    
    # Beam width statistics
    beam_widths = router.performance_metrics['beam_widths']
    print(f"\n5. BEAM WIDTHS (degrees):")
    print(f"   Min: {np.min(beam_widths):.1f}")
    print(f"   Max: {np.max(beam_widths):.1f}")
    print(f"   Mean: {np.mean(beam_widths):.2f}")
    print(f"   Median: {np.median(beam_widths):.2f}")
    print(f"   Total beam adjustments: {len(beam_widths)}")
    
    # Network statistics
    print(f"\n6. NETWORK STATISTICS:")
    print(f"   Total nodes: {num_nodes}")
    print(f"   Total packets routed: {len(router.packets)}")
    print(f"   Successful deliveries: {len([p for p in router.packets if p.delivery_time])}")
    print(f"   Network density: {sum(len(node.neighbors) for node in router.nodes.values())/len(router.nodes):.2f} neighbors per node")
    
    print("\n" + "="*50 + "\n")
    
    # Generate detailed performance report file
    report_file = router.generate_performance_report(num_nodes=num_nodes)
    print(f"Detailed performance report saved to: {report_file}")
    
    # Show metrics
    metrics_fig = router.plot_performance_metrics()
    plt.show(block=False)
    
    # Only animate the first packet routing
    if router.animation_frames:
        ani = router.animate_packet_routing(path)
        plt.show() 