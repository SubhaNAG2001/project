#!/usr/bin/env python3
# Vector-Based Forwarding (VBF) Simulation
# A 2D simulation of VBF protocol for underwater sensor networks

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path
import matplotlib.animation as animation
import random
from scipy.spatial import distance

# Set random seed for reproducibility
np.random.seed(42)

class VBFSimulation:
    def __init__(self, width=1000, height=1000, num_nodes=100, pipe_radius=100):
        self.width = width
        self.height = height
        self.num_nodes = num_nodes
        self.pipe_radius = pipe_radius
        
        # Define source and sink positions
        self.source = np.array([100, 500])
        self.sink = np.array([900, 500])
        
        # Generate random node positions
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
        
    def _find_pipe_nodes(self):
        """Find nodes that fall within the routing pipe."""
        pipe_nodes = []
        for i, node in enumerate(self.nodes):
            # Calculate distance from node to the routing vector
            proj = self._calculate_projection(node)
            dist_to_vector = self._calculate_distance_to_vector(node, proj)
            
            # Check if the node is in the pipe and between source and sink
            if (dist_to_vector <= self.pipe_radius and 
                0 <= np.dot(proj - self.source, self.unit_vector) <= np.linalg.norm(self.routing_vector)):
                pipe_nodes.append(i)
        
        return pipe_nodes
    
    def _calculate_projection(self, node):
        """Calculate the projection of a node onto the routing vector."""
        t = np.dot(node - self.source, self.unit_vector)
        return self.source + t * self.unit_vector
    
    def _calculate_distance_to_vector(self, node, projection):
        """Calculate the perpendicular distance from a node to the routing vector."""
        return np.linalg.norm(node - projection)
    
    def _calculate_forwarding_path(self):
        """Calculate the forwarding path from source to sink using the VBF protocol."""
        if not self.pipe_nodes_indices:
            return [self.source, self.sink]  # Direct path if no pipe nodes
        
        path = [self.source]
        current_node = self.source
        
        # Keep track of visited nodes to avoid cycles
        visited = set()
        
        while not np.array_equal(current_node, self.sink):
            best_node = None
            best_progress = -1
            
            for idx in self.pipe_nodes_indices:
                node = self.nodes[idx]
                
                # Skip already visited nodes
                node_tuple = tuple(node)
                if node_tuple in visited:
                    continue
                
                # Calculate progress towards the sink
                current_to_node = node - current_node
                progress = np.dot(current_to_node, self.unit_vector)
                
                # Only consider nodes that make progress towards the sink
                if progress > 0:
                    dist_to_sink = np.linalg.norm(node - self.sink)
                    
                    # Use a combination of progress and proximity to sink
                    # as the forwarding metric
                    forwarding_metric = progress / (dist_to_sink + 1)
                    
                    if forwarding_metric > best_progress:
                        best_progress = forwarding_metric
                        best_node = node
            
            # If no suitable next hop is found, go directly to sink
            if best_node is None:
                path.append(self.sink)
                break
            
            # Add the best node to the path and mark as visited
            path.append(best_node)
            visited.add(tuple(best_node))
            current_node = best_node
            
            # If we're close enough to the sink, go directly there
            if np.linalg.norm(current_node - self.sink) < self.pipe_radius / 2:
                path.append(self.sink)
                break
        
        # Make sure the sink is the last node
        if not np.array_equal(path[-1], self.sink):
            path.append(self.sink)
            
        return np.array(path)
    
    def visualize_static(self, save_path=None):
        """Create a static visualization of the VBF protocol."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Set up the plot area
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_title('Vector-Based Forwarding (VBF) Protocol Simulation')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        
        # Draw the routing pipe
        self._draw_routing_pipe(ax)
        
        # Plot all nodes
        ax.scatter(self.nodes[:, 0], self.nodes[:, 1], c='gray', alpha=0.5, label='Sensor Nodes')
        
        # Highlight pipe nodes
        ax.scatter(self.pipe_nodes[:, 0], self.pipe_nodes[:, 1], c='blue', s=80, label='Pipe Nodes')
        
        # Plot source and sink
        ax.scatter(self.source[0], self.source[1], c='green', s=150, marker='*', label='Source')
        ax.scatter(self.sink[0], self.sink[1], c='purple', s=150, marker='*', label='Sink')
        
        # Plot the forwarding path
        ax.plot(self.path[:, 0], self.path[:, 1], 'r-', lw=2, label='Forwarding Path')
        
        # Mark each hop in the path
        ax.scatter(self.path[1:-1, 0], self.path[1:-1, 1], c='red', s=120, marker='o', edgecolors='k', label='Forwarding Nodes')
        
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()
    
    def _draw_routing_pipe(self, ax):
        """Draw the routing pipe between source and sink."""
        # Create patch vertices
        vector_normal = np.array([-self.unit_vector[1], self.unit_vector[0]])
        
        top_left = self.source + self.pipe_radius * vector_normal
        top_right = self.sink + self.pipe_radius * vector_normal
        bottom_right = self.sink - self.pipe_radius * vector_normal
        bottom_left = self.source - self.pipe_radius * vector_normal
        
        pipe_vertices = [top_left, top_right, bottom_right, bottom_left, top_left]
        
        codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
        path = Path(pipe_vertices, codes)
        patch = PathPatch(path, alpha=0.15, facecolor='blue', edgecolor='blue', label='Routing Pipe')
        ax.add_patch(patch)
    
    def animate_forwarding(self, save_path=None):
        """Create an animated visualization of packet forwarding in VBF."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Set up the plot area
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_title('Vector-Based Forwarding (VBF) Protocol Animation', fontsize=14)
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        
        # Draw the routing pipe
        self._draw_routing_pipe(ax)
        
        # Plot all nodes
        all_nodes = ax.scatter(self.nodes[:, 0], self.nodes[:, 1], c='gray', alpha=0.5, label='Sensor Nodes')
        
        # Highlight pipe nodes
        pipe_nodes = ax.scatter(self.pipe_nodes[:, 0], self.pipe_nodes[:, 1], c='blue', s=80, label='Pipe Nodes')
        
        # Plot source and sink
        ax.scatter(self.source[0], self.source[1], c='green', s=150, marker='*', label='Source')
        ax.scatter(self.sink[0], self.sink[1], c='purple', s=150, marker='*', label='Sink')
        
        # Initialize the packet and path
        packet = ax.scatter([], [], s=250, c='red', marker='o', edgecolors='black', linewidths=2, 
                           zorder=10, label='Data Packet')
        path, = ax.plot([], [], 'r-', lw=3, label='Forwarding Path')
        
        # Add text annotation for packet state
        status_text = ax.text(self.width/2, 50, "", fontsize=14, 
                             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'),
                             ha='center', va='center')
        
        # Add a progress indicator
        progress_bar = ax.axhline(y=0, xmin=0, xmax=0, linewidth=8, color='green', alpha=0.7)
        
        # Init function for animation
        def init():
            packet.set_offsets(np.empty((0, 2)))
            path.set_data([], [])
            status_text.set_text("")
            progress_bar.set_xdata([0, 0])
            return packet, path, status_text, progress_bar
        
        # Animation function
        def animate(i):
            if i >= len(self.path):
                i = len(self.path) - 1
                
            # Update packet position
            packet.set_offsets(self.path[i].reshape(1, -1))
            
            # Update path
            path.set_data(self.path[:i+1, 0], self.path[:i+1, 1])
            
            # Update progress bar
            progress = i / (len(self.path) - 1) if len(self.path) > 1 else 0
            progress_bar.set_xdata([0, progress * self.width])
            
            # Update status text
            if i == 0:
                status_text.set_text("Data packet generated at source")
            elif i == len(self.path) - 1:
                status_text.set_text("Data packet reached destination!")
            else:
                hop_num = i
                prev_pos = self.path[i-1]
                current_pos = self.path[i]
                distance = np.linalg.norm(current_pos - prev_pos)
                status_text.set_text(f"Hop #{hop_num}: Moving {distance:.1f} units toward destination")
            
            return packet, path, status_text, progress_bar
        
        # Create animation with slower speed to see the packet movement
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                      frames=len(self.path) + 5, interval=1000, blit=True, repeat=True)
        
        ax.legend(loc='upper right')
        plt.tight_layout()
        
        if save_path:
            try:
                # Try using ffmpeg for better quality if available
                writer = animation.FFMpegWriter(fps=1, bitrate=1800)
                anim.save(save_path.replace('.gif', '.mp4'), writer=writer)
                print(f"Animation saved as {save_path.replace('.gif', '.mp4')}")
            except:
                # Fall back to pillow for gif
                writer = animation.PillowWriter(fps=1)
                anim.save(save_path, writer=writer)
                print(f"Animation saved as {save_path}")
        else:
            plt.show()
        
        return anim

    def visualize_step_by_step(self):
        """Create a step-by-step visualization of the forwarding path."""
        # First create the overall static visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Set up the plot area
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_title('Vector-Based Forwarding (VBF) Protocol - Initial Network')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        
        # Draw the routing pipe
        self._draw_routing_pipe(ax)
        
        # Plot all nodes
        ax.scatter(self.nodes[:, 0], self.nodes[:, 1], c='gray', alpha=0.5, label='Sensor Nodes')
        
        # Highlight pipe nodes
        ax.scatter(self.pipe_nodes[:, 0], self.pipe_nodes[:, 1], c='blue', s=80, label='Pipe Nodes')
        
        # Plot source and sink
        ax.scatter(self.source[0], self.source[1], c='green', s=150, marker='*', label='Source')
        ax.scatter(self.sink[0], self.sink[1], c='purple', s=150, marker='*', label='Sink')
        
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.show()
        
        # Show step-by-step packet forwarding
        for i in range(len(self.path)):
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Set up the plot area
            ax.set_xlim(0, self.width)
            ax.set_ylim(0, self.height)
            if i == 0:
                title_text = "Step 1: Data packet generated at source"
            elif i == len(self.path) - 1:
                title_text = f"Step {i+1}: Packet reached destination!"
            else:
                title_text = f"Step {i+1}: Forwarding via hop #{i}"
            
            ax.set_title(title_text)
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            
            # Draw the routing pipe
            self._draw_routing_pipe(ax)
            
            # Plot all nodes
            ax.scatter(self.nodes[:, 0], self.nodes[:, 1], c='gray', alpha=0.5, label='Sensor Nodes')
            
            # Highlight pipe nodes
            ax.scatter(self.pipe_nodes[:, 0], self.pipe_nodes[:, 1], c='blue', s=80, label='Pipe Nodes')
            
            # Plot source and sink
            ax.scatter(self.source[0], self.source[1], c='green', s=150, marker='*', label='Source')
            ax.scatter(self.sink[0], self.sink[1], c='purple', s=150, marker='*', label='Sink')
            
            # Plot the partial path up to current position
            ax.plot(self.path[:i+1, 0], self.path[:i+1, 1], 'r-', lw=2, label='Forwarding Path')
            
            # Mark the nodes in the path
            if i > 0:
                ax.scatter(self.path[1:i, 0], self.path[1:i, 1], c='red', s=120, marker='o', 
                          edgecolors='k', label='Previous Hops')
            
            # Highlight the current packet position
            ax.scatter(self.path[i, 0], self.path[i, 1], c='yellow', s=180, marker='o', 
                      edgecolors='red', linewidths=2, zorder=10, label='Current Packet Position')
            
            ax.legend(loc='upper right')
            plt.tight_layout()
            plt.savefig(f'step_{i+1}.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Wait for user to press enter to continue to next step
            input(f"Press Enter to continue to step {i+2 if i+1 < len(self.path) else 'completion'}...")

    def _calculate_performance_metrics(self):
        """Calculate energy consumption, transmission delay, and other metrics."""
        # Energy model parameters (simplified)
        transmit_energy_per_bit = 0.5  # energy to transmit 1 bit (in microjoules)
        receive_energy_per_bit = 0.25  # energy to receive 1 bit (in microjoules)
        packet_size = 1000  # bits
        
        # Transmission model parameters
        propagation_speed = 1500  # meters/second in water
        data_rate = 10000  # bits/second
        
        # Calculate energy and delay for each hop
        total_energy = 0
        total_delay = 0
        hop_delays = []
        
        for i in range(len(self.path) - 1):
            # Distance for this hop
            hop_distance = np.linalg.norm(self.path[i+1] - self.path[i])
            
            # Transmission delay components
            transmission_time = packet_size / data_rate  # time to transmit the packet
            propagation_time = hop_distance / propagation_speed  # time for signal to travel
            processing_delay = 0.01  # seconds (fixed processing overhead)
            
            # Total delay for this hop
            hop_delay = transmission_time + propagation_time + processing_delay
            hop_delays.append(hop_delay)
            total_delay += hop_delay
            
            # Energy for this hop
            tx_energy = transmit_energy_per_bit * packet_size
            rx_energy = receive_energy_per_bit * packet_size
            hop_energy = tx_energy + rx_energy
            
            # Add distance-based component (energy increases with square of distance)
            hop_energy *= (1 + (hop_distance / 100) ** 2)
            
            total_energy += hop_energy
        
        # Store metrics
        self.energy_consumption = total_energy
        self.transmission_delays = hop_delays
        self.total_delay = total_delay
        
        # Simulate packet delivery ratio based on path length
        # (longer paths have higher chance of failure)
        base_success_rate = 0.99
        hop_failure_probability = 0.01
        success_probability = base_success_rate * (1 - hop_failure_probability) ** (len(self.path) - 1)
        self.packet_delivery_ratio = success_probability

    def display_metrics_terminal(self):
        """Display performance metrics in the terminal."""
        print("\n===== VBF PROTOCOL PERFORMANCE METRICS =====")
        print(f"Pipe Radius: {self.pipe_radius} units")
        print(f"Total Nodes: {self.num_nodes}")
        print(f"Nodes in Pipe: {len(self.pipe_nodes_indices)} ({len(self.pipe_nodes_indices)/self.num_nodes*100:.1f}%)")
        print(f"Source-to-Sink Distance: {np.linalg.norm(self.sink - self.source):.1f} units")
        print(f"Hop Count: {self.hop_count}")
        print(f"Path Length: {self._calculate_path_length():.1f} units")
        print("\n--- Energy ---")
        print(f"Total Energy Consumption: {self.energy_consumption:.2f} microjoules")
        print(f"Energy per Hop: {self.energy_consumption/max(1, self.hop_count):.2f} microjoules/hop")
        print("\n--- Delay ---")
        print(f"Total End-to-End Delay: {self.total_delay:.4f} seconds")
        avg_hop_delay = self.total_delay / max(1, self.hop_count)
        print(f"Average Delay per Hop: {avg_hop_delay:.4f} seconds/hop")
        print("\n--- Reliability ---")
        print(f"Packet Delivery Ratio: {self.packet_delivery_ratio:.4f}")
        print("===========================================\n")
        
    def save_metrics_to_file(self, filename="vbf_metrics.txt"):
        """Save performance metrics to a text file."""
        with open(filename, 'w') as f:
            f.write("===== VBF PROTOCOL PERFORMANCE METRICS =====\n")
            f.write(f"Pipe Radius: {self.pipe_radius} units\n")
            f.write(f"Total Nodes: {self.num_nodes}\n")
            f.write(f"Nodes in Pipe: {len(self.pipe_nodes_indices)} ({len(self.pipe_nodes_indices)/self.num_nodes*100:.1f}%)\n")
            f.write(f"Source-to-Sink Distance: {np.linalg.norm(self.sink - self.source):.1f} units\n")
            f.write(f"Hop Count: {self.hop_count}\n")
            f.write(f"Path Length: {self._calculate_path_length():.1f} units\n")
            f.write("\n--- Energy ---\n")
            f.write(f"Total Energy Consumption: {self.energy_consumption:.2f} microjoules\n")
            f.write(f"Energy per Hop: {self.energy_consumption/max(1, self.hop_count):.2f} microjoules/hop\n")
            f.write("\n--- Delay ---\n")
            f.write(f"Total End-to-End Delay: {self.total_delay:.4f} seconds\n")
            avg_hop_delay = self.total_delay / max(1, self.hop_count)
            f.write(f"Average Delay per Hop: {avg_hop_delay:.4f} seconds/hop\n")
            f.write("\n--- Reliability ---\n")
            f.write(f"Packet Delivery Ratio: {self.packet_delivery_ratio:.4f}\n")
            
            # Add detailed hop-by-hop metrics
            f.write("\n===== HOP-BY-HOP METRICS =====\n")
            for i in range(len(self.path) - 1):
                hop_distance = np.linalg.norm(self.path[i+1] - self.path[i])
                f.write(f"\nHop {i+1}:\n")
                f.write(f"  Distance: {hop_distance:.2f} units\n")
                f.write(f"  Delay: {self.transmission_delays[i]:.4f} seconds\n")
                
                # Calculate energy for this hop
                transmit_energy = 0.5 * 1000  # transmit_energy_per_bit * packet_size
                receive_energy = 0.25 * 1000  # receive_energy_per_bit * packet_size
                hop_energy = (transmit_energy + receive_energy) * (1 + (hop_distance / 100) ** 2)
                f.write(f"  Energy: {hop_energy:.2f} microjoules\n")
            
            f.write("\n===== COMPARATIVE ANALYSIS =====\n")
            f.write("Effect of pipe radius on performance metrics:\n\n")
            
            # Add comparative data for different pipe radii
            pipe_radii = [50, 100, 150, 200, 250]
            f.write("Pipe Radius | Hop Count | Path Length | Energy | Delay | PDR\n")
            f.write("--------------------------------------------------------\n")
            
            # Store current pipe radius
            original_radius = self.pipe_radius
            
            for radius in pipe_radii:
                # Reset the simulation with this radius
                self.pipe_radius = radius
                self.pipe_nodes_indices = self._find_pipe_nodes()
                self.pipe_nodes = self.nodes[self.pipe_nodes_indices]
                self.path = self._calculate_forwarding_path()
                self.hop_count = len(self.path) - 1
                self._calculate_performance_metrics()
                
                # Write the metrics
                path_length = self._calculate_path_length()
                f.write(f"{radius:11d} | {self.hop_count:9d} | {path_length:11.1f} | {self.energy_consumption:6.0f} | {self.total_delay:5.3f} | {self.packet_delivery_ratio:.4f}\n")
            
            # Restore original radius
            self.pipe_radius = original_radius
            self.pipe_nodes_indices = self._find_pipe_nodes()
            self.pipe_nodes = self.nodes[self.pipe_nodes_indices]
            self.path = self._calculate_forwarding_path()
            self.hop_count = len(self.path) - 1
            self._calculate_performance_metrics()
            
            f.write("\n===========================================\n")
            print(f"Performance metrics saved to {filename}")

    def _calculate_path_length(self):
        """Calculate the total path length."""
        length = 0
        for i in range(len(self.path) - 1):
            length += np.linalg.norm(self.path[i+1] - self.path[i])
        return length

    def plot_performance_metrics(self):
        """Create plots for energy, delay, and packet delivery metrics with improved accuracy."""
        plt.style.use('ggplot')  # Use a professional-looking style
        
        # Create a figure with 3 subplots in a 2x2 grid for better readability
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(2, 2)
        
        # Add a title to the entire figure
        fig.suptitle('Vector-Based Forwarding (VBF) Protocol Performance Metrics', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Plot 1: Energy and delay per hop (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        hop_indices = list(range(1, len(self.transmission_delays) + 1))
        
        # Create a twin axis for delay
        ax1_twin = ax1.twinx()
        
        # Calculate energy per hop with more accurate model
        hop_energies = []
        for i in range(len(self.path) - 1):
            hop_distance = np.linalg.norm(self.path[i+1] - self.path[i])
            # More accurate energy model
            transmit_energy = 0.5 * 1000  # transmit_energy_per_bit * packet_size
            receive_energy = 0.25 * 1000  # receive_energy_per_bit * packet_size
            # Add distance-based component with a more realistic quadratic model
            transmission_power_factor = (hop_distance / 100) ** 2
            hop_energy = (transmit_energy + receive_energy) * (1 + transmission_power_factor)
            hop_energies.append(hop_energy)
        
        # Plot energy bars with improved style
        bars = ax1.bar(hop_indices, hop_energies, width=0.6, alpha=0.7, 
                      color='#3274A1', edgecolor='black', linewidth=1.2, label='Energy')
        
        # Add energy values on top of bars
        for bar, energy in zip(bars, hop_energies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 100,
                    f'{energy:.0f}',
                    ha='center', va='bottom', fontsize=9, rotation=0)
        
        # Plot delay line with improved style
        line = ax1_twin.plot(hop_indices, self.transmission_delays, 'ro-', 
                            linewidth=2.5, markersize=8, label='Delay')
        
        # Add delay values
        for i, delay in enumerate(self.transmission_delays):
            ax1_twin.text(hop_indices[i], delay + 0.02, f'{delay:.3f}s', 
                         ha='center', va='bottom', fontsize=9, color='red')
        
        ax1.set_xlabel('Hop Number', fontsize=12)
        ax1.set_ylabel('Energy Consumption (microjoules)', fontsize=12, color='#3274A1')
        ax1_twin.set_ylabel('Transmission Delay (seconds)', fontsize=12, color='red')
        ax1.tick_params(axis='y', labelcolor='#3274A1')
        ax1_twin.tick_params(axis='y', labelcolor='red')
        ax1.set_title('Energy and Delay per Hop', fontsize=14, pad=10)
        
        # Set the x-axis to show integer hop numbers
        ax1.set_xticks(hop_indices)
        ax1.set_xticklabels([str(i) for i in hop_indices])
        
        # Add both legends with better positioning
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left', frameon=True, fontsize=10)
        
        # Plot 2: Cumulative energy and delay (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        cumulative_energy = np.cumsum(hop_energies)
        cumulative_delay = np.cumsum(self.transmission_delays)
        
        ax2_twin = ax2.twinx()
        
        # Plot cumulative metrics with improved style
        ax2.plot(hop_indices, cumulative_energy, 'b-o', 
                linewidth=2.5, markersize=8, label='Cumulative Energy')
        ax2_twin.plot(hop_indices, cumulative_delay, 'r-o', 
                     linewidth=2.5, markersize=8, label='Cumulative Delay')
        
        # Add data labels to the last point
        ax2.text(hop_indices[-1], cumulative_energy[-1] + 100, 
                f'{cumulative_energy[-1]:.0f}', color='blue', fontsize=10)
        ax2_twin.text(hop_indices[-1], cumulative_delay[-1] + 0.02, 
                     f'{cumulative_delay[-1]:.3f}s', color='red', fontsize=10)
        
        ax2.set_xlabel('Hop Number', fontsize=12)
        ax2.set_ylabel('Cumulative Energy (microjoules)', fontsize=12, color='blue')
        ax2_twin.set_ylabel('Cumulative Delay (seconds)', fontsize=12, color='red')
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2_twin.tick_params(axis='y', labelcolor='red')
        ax2.set_title('Cumulative Energy and Delay', fontsize=14, pad=10)
        
        # Set the x-axis to show integer hop numbers
        ax2.set_xticks(hop_indices)
        ax2.set_xticklabels([str(i) for i in hop_indices])
        
        # Add both legends with better positioning
        lines, labels = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left', frameon=True, fontsize=10)
        
        # Plot 3: Path efficiency metrics (bottom left)
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Calculate metrics for different pipe radii with more datapoints
        pipe_radii = [50, 75, 100, 125, 150, 175, 200, 225, 250]
        hop_counts = []
        path_lengths = []
        energy_values = []
        delay_values = []
        pdr_values = []
        
        # Store current pipe radius
        original_radius = self.pipe_radius
        
        for radius in pipe_radii:
            # Reset the simulation with this radius
            self.pipe_radius = radius
            self.pipe_nodes_indices = self._find_pipe_nodes()
            self.pipe_nodes = self.nodes[self.pipe_nodes_indices]
            self.path = self._calculate_forwarding_path()
            self.hop_count = len(self.path) - 1
            self._calculate_performance_metrics()
            
            # Store the metrics
            hop_counts.append(self.hop_count)
            path_lengths.append(self._calculate_path_length())
            energy_values.append(self.energy_consumption)
            delay_values.append(self.total_delay)
            pdr_values.append(self.packet_delivery_ratio)
        
        # Restore original radius
        self.pipe_radius = original_radius
        self.pipe_nodes_indices = self._find_pipe_nodes()
        self.pipe_nodes = self.nodes[self.pipe_nodes_indices]
        self.path = self._calculate_forwarding_path()
        self.hop_count = len(self.path) - 1
        self._calculate_performance_metrics()
        
        # X-axis positions for the bar chart
        x = np.arange(len(pipe_radii))
        width = 0.2
        
        # Plot bars for each metric with improved colors and style
        colors = ['#3274A1', '#E1812C', '#3A923A', '#C03D3E']
        bars1 = ax3.bar(x - width*1.5, [h/max(hop_counts) for h in hop_counts], width, 
                       color=colors[0], alpha=0.8, edgecolor='black', linewidth=1, label='Hop Count')
        bars2 = ax3.bar(x - width/2, [e/max(energy_values) for e in energy_values], width, 
                       color=colors[1], alpha=0.8, edgecolor='black', linewidth=1, label='Energy')
        bars3 = ax3.bar(x + width/2, [d/max(delay_values) for d in delay_values], width, 
                       color=colors[2], alpha=0.8, edgecolor='black', linewidth=1, label='Delay')
        bars4 = ax3.bar(x + width*1.5, pdr_values, width, 
                       color=colors[3], alpha=0.8, edgecolor='black', linewidth=1, label='PDR')
        
        ax3.set_xlabel('Pipe Radius', fontsize=12)
        ax3.set_ylabel('Normalized Value', fontsize=12)
        ax3.set_title('Effect of Pipe Radius on Performance Metrics', fontsize=14, pad=10)
        ax3.set_xticks(x)
        ax3.set_xticklabels(pipe_radii)
        ax3.legend(frameon=True, fontsize=10)
        
        # Add a grid for better readability
        ax3.grid(True, linestyle='--', alpha=0.6)
        
        # Plot 4: Raw metrics vs pipe radius (bottom right)
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Plot the non-normalized values with individual y-axes
        ax4.plot(pipe_radii, hop_counts, 'o-', color=colors[0], linewidth=2, label='Hop Count')
        ax4.set_xlabel('Pipe Radius', fontsize=12)
        ax4.set_ylabel('Hop Count', fontsize=12, color=colors[0])
        ax4.tick_params(axis='y', labelcolor=colors[0])
        
        # Create additional y-axes
        ax4_2 = ax4.twinx()
        ax4_3 = ax4.twinx()
        ax4_3.spines['right'].set_position(('outward', 60))
        
        # Plot energy and delay
        ax4_2.plot(pipe_radii, energy_values, 's-', color=colors[1], linewidth=2, label='Energy')
        ax4_3.plot(pipe_radii, delay_values, '^-', color=colors[2], linewidth=2, label='Delay')
        
        ax4_2.set_ylabel('Energy Consumption (microjoules)', fontsize=12, color=colors[1])
        ax4_3.set_ylabel('End-to-End Delay (seconds)', fontsize=12, color=colors[2])
        
        ax4_2.tick_params(axis='y', labelcolor=colors[1])
        ax4_3.tick_params(axis='y', labelcolor=colors[2])
        
        # Add PDR on a secondary axis with percentage scale
        ax4_4 = ax4.twinx()
        ax4_4.spines['right'].set_position(('outward', 120))
        ax4_4.plot(pipe_radii, [pdr * 100 for pdr in pdr_values], 'd-', color=colors[3], linewidth=2, label='PDR')
        ax4_4.set_ylabel('Packet Delivery Ratio (%)', fontsize=12, color=colors[3])
        ax4_4.tick_params(axis='y', labelcolor=colors[3])
        ax4_4.set_ylim([95, 100])  # Set a more useful range for PDR percentage
        
        # Combined legend
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_2.get_legend_handles_labels()
        lines3, labels3 = ax4_3.get_legend_handles_labels()
        lines4, labels4 = ax4_4.get_legend_handles_labels()
        
        ax4.legend(lines1 + lines2 + lines3 + lines4, 
                 labels1 + labels2 + labels3 + labels4, 
                 loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                 ncol=4, frameon=True, fontsize=10)
        
        ax4.set_title('Performance Metrics vs. Pipe Radius', fontsize=14, pad=10)
        ax4.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.3)
        plt.savefig('vbf_performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()

def compare_pipe_radius():
    """Compare the effect of different pipe radii on the forwarding path."""
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    
    # Standard VBF with moderate pipe radius
    sim1 = VBFSimulation(pipe_radius=100)
    
    # Draw in the first subplot
    ax1 = axs[0]
    ax1.set_xlim(0, sim1.width)
    ax1.set_ylim(0, sim1.height)
    ax1.set_title('Standard VBF (Pipe Radius = 100)')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    
    # Draw the routing pipe
    sim1._draw_routing_pipe(ax1)
    
    # Plot all nodes
    ax1.scatter(sim1.nodes[:, 0], sim1.nodes[:, 1], c='gray', alpha=0.5)
    
    # Highlight pipe nodes
    ax1.scatter(sim1.pipe_nodes[:, 0], sim1.pipe_nodes[:, 1], c='blue', s=80)
    
    # Plot source and sink
    ax1.scatter(sim1.source[0], sim1.source[1], c='green', s=150, marker='*')
    ax1.scatter(sim1.sink[0], sim1.sink[1], c='purple', s=150, marker='*')
    
    # Plot the forwarding path
    ax1.plot(sim1.path[:, 0], sim1.path[:, 1], 'r-', lw=2)
    
    # Mark each hop in the path
    ax1.scatter(sim1.path[1:-1, 0], sim1.path[1:-1, 1], c='red', s=120, marker='o', edgecolors='k')
    
    # Adaptive VBF with larger pipe radius
    sim2 = VBFSimulation(pipe_radius=200)
    
    # Draw in the second subplot
    ax2 = axs[1]
    ax2.set_xlim(0, sim2.width)
    ax2.set_ylim(0, sim2.height)
    ax2.set_title('Adaptive VBF (Pipe Radius = 200)')
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    
    # Draw the routing pipe
    sim2._draw_routing_pipe(ax2)
    
    # Plot all nodes
    ax2.scatter(sim2.nodes[:, 0], sim2.nodes[:, 1], c='gray', alpha=0.5)
    
    # Highlight pipe nodes
    ax2.scatter(sim2.pipe_nodes[:, 0], sim2.pipe_nodes[:, 1], c='blue', s=80)
    
    # Plot source and sink
    ax2.scatter(sim2.source[0], sim2.source[1], c='green', s=150, marker='*')
    ax2.scatter(sim2.sink[0], sim2.sink[1], c='purple', s=150, marker='*')
    
    # Plot the forwarding path
    ax2.plot(sim2.path[:, 0], sim2.path[:, 1], 'r-', lw=2)
    
    # Mark each hop in the path
    ax2.scatter(sim2.path[1:-1, 0], sim2.path[1:-1, 1], c='red', s=120, marker='o', edgecolors='k')
    
    plt.tight_layout()
    plt.savefig('standard_vbf.png', dpi=300, bbox_inches='tight')
    plt.savefig('adaptive_vbf.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Create a simulation with custom settings
    sim = VBFSimulation(width=1000, height=1000, num_nodes=150, pipe_radius=120)
    
    # First show the animation of data packet transmission
    print("Displaying data packet transmission animation...")
    sim.animate_forwarding()  # This will show the animation in a window
    
    # After watching the animation, display performance metrics in terminal
    sim.display_metrics_terminal()
    
    # Save metrics to file
    sim.save_metrics_to_file()
    
    # Plot improved performance metrics
    sim.plot_performance_metrics()
    
    # Skip comparison to focus on animation and metrics
    # compare_pipe_radius() 