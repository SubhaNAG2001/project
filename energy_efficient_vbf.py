#!/usr/bin/env python3
# Energy Efficient Vector-Based Forwarding (E-VBF) Simulation
# An energy-optimized version of the standard VBF protocol for underwater sensor networks

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path
import matplotlib.animation as animation
import random
from scipy.spatial import distance
import copy

# Set random seed for reproducibility
np.random.seed(42)

class EnergyEfficientVBF:
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
        
        # Initialize node energy levels (100 units for each node)
        self.energy_levels = np.ones(num_nodes) * 100
        
        # Adaptive transmission power levels (percentage of max power)
        self.power_levels = [0.2, 0.4, 0.6, 0.8, 1.0]
        
        # Duty cycling parameters (percent of time active)
        self.duty_cycle = 0.5  # 50% duty cycle by default
        self.active_nodes = np.random.choice([True, False], num_nodes, p=[self.duty_cycle, 1-self.duty_cycle])
        
        # Residual energy threshold for participation (percent)
        self.energy_threshold = 0.2  # Nodes with < 20% energy won't forward packets
        
        # Vector from source to sink
        self.routing_vector = self.sink - self.source
        self.unit_vector = self.routing_vector / np.linalg.norm(self.routing_vector)
        
        # Identify nodes in the routing pipe
        self.pipe_nodes_indices = self._find_pipe_nodes()
        self.pipe_nodes = self.nodes[self.pipe_nodes_indices]
        
        # Find forwarding path with energy awareness
        self.path = self._calculate_energy_efficient_path()
        
        # Performance metrics
        self.energy_consumption = 0
        self.transmission_delays = []  # In milliseconds
        self.total_delay = 0  # In milliseconds
        self.packet_delivery_ratio = 0
        self.hop_count = len(self.path) - 1
        self.throughput = 0  # Added throughput metric
        self.energy_efficiency = 0  # bits/μJ
        self.power_levels_used = []  # Track which power levels were used
        
        # Calculate performance metrics
        self._calculate_performance_metrics()
        
    def _find_pipe_nodes(self):
        """Find nodes that fall within the routing pipe and have sufficient energy."""
        pipe_nodes = []
        for i, node in enumerate(self.nodes):
            # Check if node is active in the current duty cycle
            if not self.active_nodes[i]:
                continue
                
            # Check if node has sufficient energy
            if self.energy_levels[i] < 100 * self.energy_threshold:
                continue
                
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
    
    def _minimum_required_power_level(self, node1, node2):
        """Determine minimum power level needed for communication between two nodes."""
        distance = np.linalg.norm(node1 - node2)
        
        # Power required increases with square of distance
        # Define thresholds for different power levels
        if distance < 200:
            return self.power_levels[0]  # 20% power for short distances
        elif distance < 300:
            return self.power_levels[1]  # 40% power
        elif distance < 400:
            return self.power_levels[2]  # 60% power
        elif distance < 500:
            return self.power_levels[3]  # 80% power
        else:
            return self.power_levels[4]  # 100% power for long distances
    
    def _energy_consumption_for_transmission(self, distance, power_level, packet_size=1000):
        """Calculate energy consumed for a transmission based on distance and power level."""
        base_transmit_energy = 0.5 * packet_size  # Base energy for sending 1000 bits
        base_receive_energy = 0.25 * packet_size  # Base energy for receiving
        
        # Energy consumption scales with power level and distance
        transmission_energy = base_transmit_energy * power_level * (1 + (distance / 100) ** 2)
        receive_energy = base_receive_energy
        
        return transmission_energy + receive_energy
    
    def _node_desirability(self, current_node, candidate_node_idx):
        """Calculate desirability of a node as next hop based on multiple factors."""
        candidate_node = self.nodes[candidate_node_idx]
        
        # Factor 1: Progress towards sink (primary goal)
        progress = np.dot(candidate_node - current_node, self.unit_vector)
        if progress <= 0:
            return -1  # Negative progress is undesirable
        
        # Factor 2: Distance to sink
        dist_to_sink = np.linalg.norm(candidate_node - self.sink)
        dist_factor = 1.0 / (1.0 + dist_to_sink / 100)  # Normalize
        
        # Factor 3: Energy level of the candidate node
        energy_factor = self.energy_levels[candidate_node_idx] / 100
        
        # Factor 4: Power required for transmission
        power_level = self._minimum_required_power_level(current_node, candidate_node)
        power_factor = 1.0 - power_level  # Lower power is better
        
        # Weighted combination of factors 
        # Progress is most important, followed by energy, then power, then distance
        return 0.4 * progress + 0.3 * energy_factor + 0.2 * power_factor + 0.1 * dist_factor
    
    def _calculate_energy_efficient_path(self):
        """Calculate the most energy-efficient forwarding path from source to sink."""
        if not self.pipe_nodes_indices:
            return np.array([self.source, self.sink])  # Direct path if no pipe nodes
        
        path = [self.source]
        current_node = self.source
        power_used = []
        
        # Keep track of visited nodes to avoid cycles
        visited = set()
        
        while not np.array_equal(current_node, self.sink):
            best_node = None
            best_desirability = -1
            best_power_level = 1.0
            
            for idx in self.pipe_nodes_indices:
                node = self.nodes[idx]
                
                # Skip already visited nodes
                node_tuple = tuple(node)
                if node_tuple in visited:
                    continue
                
                # Calculate node desirability
                desirability = self._node_desirability(current_node, idx)
                
                if desirability > best_desirability:
                    best_desirability = desirability
                    best_node = node
                    best_power_level = self._minimum_required_power_level(current_node, node)
            
            # If no suitable next hop is found, try direct transmission to sink
            if best_node is None:
                # Check if direct transmission to sink is possible with maximum power
                if np.linalg.norm(current_node - self.sink) <= 500:  # Max transmission range
                    power_used.append(1.0)  # Use maximum power
                    path.append(self.sink)
                    break
                else:
                    # No forwarding path available
                    return np.array([self.source, self.sink])
            
            # Add the best node to the path and mark as visited
            path.append(best_node)
            power_used.append(best_power_level)
            visited.add(tuple(best_node))
            current_node = best_node
            
            # If we're close enough to the sink, go directly there
            if np.linalg.norm(current_node - self.sink) < 200:  # Short distance
                power_level = self._minimum_required_power_level(current_node, self.sink)
                power_used.append(power_level)
                path.append(self.sink)
                break
        
        # Store the power levels used for each hop
        self.power_levels_used = power_used
        
        # Make sure the sink is the last node
        if not np.array_equal(path[-1], self.sink):
            power_level = self._minimum_required_power_level(path[-1], self.sink)
            power_used.append(power_level)
            path.append(self.sink)
            
        return np.array(path)
    
    def _calculate_performance_metrics(self):
        """Calculate energy consumption, transmission delay, and other metrics."""
        # Packet parameters
        packet_size = 1000  # bits
        
        # Transmission model parameters
        propagation_speed = 1500  # meters/second in water
        data_rate = 10000  # bits/second
        
        # Calculate energy and delay for each hop
        total_energy = 0
        total_delay_ms = 0
        hop_delays_ms = []
        
        for i in range(len(self.path) - 1):
            # Distance for this hop
            hop_distance = np.linalg.norm(self.path[i+1] - self.path[i])
            
            # Power level used for this hop
            power_level = self.power_levels_used[i] if i < len(self.power_levels_used) else 1.0
            
            # Transmission delay components (in seconds)
            transmission_time = packet_size / data_rate  # time to transmit the packet
            propagation_time = hop_distance / propagation_speed  # time for signal to travel
            processing_delay = 0.01  # seconds (fixed processing overhead)
            
            # Total delay for this hop (convert to milliseconds)
            hop_delay_sec = transmission_time + propagation_time + processing_delay
            hop_delay_ms = hop_delay_sec * 1000  # Convert to milliseconds
            hop_delays_ms.append(hop_delay_ms)
            total_delay_ms += hop_delay_ms
            
            # Energy for this hop using the adaptive power level
            hop_energy = self._energy_consumption_for_transmission(hop_distance, power_level, packet_size)
            total_energy += hop_energy
        
        # Store metrics
        self.energy_consumption = total_energy
        self.transmission_delays = hop_delays_ms
        self.total_delay = total_delay_ms
        
        # Calculate throughput (bits per second)
        if total_delay_ms > 0:
            self.throughput = (packet_size * self.hop_count) / (total_delay_ms / 1000.0)
        else:
            self.throughput = 0
            
        # Calculate energy efficiency (bits transmitted per microjoule)
        if total_energy > 0:
            self.energy_efficiency = (packet_size * self.hop_count) / total_energy
        else:
            self.energy_efficiency = 0
        
        # Simulate packet delivery ratio based on path length and power levels
        base_success_rate = 0.99
        hop_failure_probability = 0.01
        # Adjust failure probability based on power levels (lower power might have higher failure)
        adjusted_failure_probs = [0.01 + (1 - power) * 0.02 for power in self.power_levels_used]
        
        # Calculate overall PDR
        success_probability = base_success_rate
        for i in range(len(self.path) - 1):
            if i < len(adjusted_failure_probs):
                success_probability *= (1 - adjusted_failure_probs[i])
            else:
                success_probability *= (1 - hop_failure_probability)
                
        self.packet_delivery_ratio = success_probability
    
    def visualize_static(self, compare_with_standard=False, standard_vbf_path=None, save_path=None):
        """Create a static visualization of the energy-efficient VBF protocol."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Set up the plot area
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_title('Energy-Efficient Vector-Based Forwarding (E-VBF) Protocol', fontsize=14)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        
        # Draw the routing pipe
        self._draw_routing_pipe(ax)
        
        # Plot all nodes with color based on energy level
        energy_colors = []
        for i in range(len(self.nodes)):
            if i in self.pipe_nodes_indices:
                # More blue for higher energy nodes in pipe
                energy_normalized = self.energy_levels[i] / 100
                energy_colors.append((0, 0, min(1, 0.5 + energy_normalized/2)))
            else:
                # Gray for nodes outside pipe
                energy_colors.append((0.7, 0.7, 0.7, 0.5))
        
        # Plot all nodes
        ax.scatter(self.nodes[:, 0], self.nodes[:, 1], c=energy_colors, alpha=0.5, label='Sensor Nodes')
        
        # Highlight pipe nodes differently - size based on energy
        sizes = [30 + (self.energy_levels[i]/100) * 70 for i in self.pipe_nodes_indices]
        ax.scatter(self.pipe_nodes[:, 0], self.pipe_nodes[:, 1], c='blue', s=sizes, label='Pipe Nodes')
        
        # Plot source and sink
        ax.scatter(self.source[0], self.source[1], c='green', s=150, marker='*', label='Source')
        ax.scatter(self.sink[0], self.sink[1], c='purple', s=150, marker='*', label='Sink')
        
        # Plot the energy-efficient forwarding path
        ax.plot(self.path[:, 0], self.path[:, 1], 'r-', lw=2, label='E-VBF Path')
        
        # Mark each hop in the path with size based on power level used
        for i in range(1, len(self.path) - 1):
            power_idx = i - 1
            power_level = self.power_levels_used[power_idx] if power_idx < len(self.power_levels_used) else 1.0
            node_size = 80 + power_level * 100  # Size based on power
            ax.scatter(self.path[i, 0], self.path[i, 1], c='red', s=node_size, marker='o', 
                      edgecolors='k', label='_' if i > 1 else 'Forwarding Nodes')
            
            # Add power level annotation
            ax.annotate(f"{int(power_level*100)}%", 
                      (self.path[i, 0], self.path[i, 1]), 
                      xytext=(10, 10), textcoords='offset points',
                      bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        # Compare with standard VBF if requested
        if compare_with_standard and standard_vbf_path is not None:
            ax.plot(standard_vbf_path[:, 0], standard_vbf_path[:, 1], 'b--', lw=2, label='Standard VBF Path')
            # Add legend for comparison
            ax.legend(loc='upper right')
            ax.set_title('Energy-Efficient VBF vs. Standard VBF Comparison', fontsize=14)
        else:
            ax.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
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
    
    def display_metrics_terminal(self):
        """Display performance metrics in the terminal."""
        print("\n===== ENERGY-EFFICIENT VBF PROTOCOL PERFORMANCE METRICS =====")
        print(f"Pipe Radius: {self.pipe_radius} units")
        print(f"Total Nodes: {self.num_nodes}")
        print(f"Nodes in Pipe: {len(self.pipe_nodes_indices)} ({len(self.pipe_nodes_indices)/self.num_nodes*100:.1f}%)")
        print(f"Source-to-Sink Distance: {np.linalg.norm(self.sink - self.source):.1f} units")
        print(f"Hop Count: {self.hop_count}")
        print(f"Path Length: {self._calculate_path_length():.1f} units")
        print("\n--- Energy ---")
        print(f"Total Energy Consumption: {self.energy_consumption:.2f} microjoules")
        print(f"Energy per Hop: {self.energy_consumption/max(1, self.hop_count):.2f} microjoules/hop")
        print(f"Energy Efficiency: {self.energy_efficiency:.4f} bits/microjoule")
        print("\n--- Duty Cycling ---")
        print(f"Duty Cycle: {self.duty_cycle*100:.1f}% (nodes active {self.duty_cycle*100:.1f}% of time)")
        print(f"Energy Threshold: {self.energy_threshold*100:.1f}% (minimum energy to participate)")
        print("\n--- Adaptive Power Levels ---")
        
        if self.power_levels_used:
            avg_power = sum(self.power_levels_used) / len(self.power_levels_used)
            print(f"Average Power Level Used: {avg_power*100:.1f}%")
            power_distribution = [f"{power*100:.0f}%" for power in self.power_levels_used]
            print(f"Power Levels by Hop: {', '.join(power_distribution)}")
        
        print("\n--- Delay ---")
        print(f"Total End-to-End Delay: {self.total_delay:.2f} milliseconds")
        avg_hop_delay = self.total_delay / max(1, self.hop_count)
        print(f"Average Delay per Hop: {avg_hop_delay:.2f} milliseconds/hop")
        print("\n--- Throughput ---")
        print(f"Network Throughput: {self.throughput:.2f} bits/second")
        print("\n--- Reliability ---")
        print(f"Packet Delivery Ratio: {self.packet_delivery_ratio:.4f}")
        print("===========================================\n")
    
    def _calculate_path_length(self):
        """Calculate the total path length."""
        length = 0
        for i in range(len(self.path) - 1):
            length += np.linalg.norm(self.path[i+1] - self.path[i])
        return length
    
    def compare_with_standard_vbf(self, standard_vbf):
        """Compare energy-efficient VBF with standard VBF."""
        print("\n===== ENERGY EFFICIENCY COMPARISON =====")
        print(f"{'Metric':<30} {'Standard VBF':<15} {'Energy-Efficient VBF':<20} {'Improvement':<10}")
        print("-" * 80)
        
        # Energy metrics
        energy_improvement = (1 - self.energy_consumption / standard_vbf.energy_consumption) * 100
        print(f"{'Energy Consumption (μJ)':<30} {standard_vbf.energy_consumption:<15.2f} {self.energy_consumption:<20.2f} {energy_improvement:>9.1f}%")
        
        # Energy per hop
        std_energy_per_hop = standard_vbf.energy_consumption / max(1, standard_vbf.hop_count)
        e_energy_per_hop = self.energy_consumption / max(1, self.hop_count)
        energy_per_hop_improvement = (1 - e_energy_per_hop / std_energy_per_hop) * 100
        print(f"{'Energy per Hop (μJ/hop)':<30} {std_energy_per_hop:<15.2f} {e_energy_per_hop:<20.2f} {energy_per_hop_improvement:>9.1f}%")
        
        # Energy efficiency
        efficiency_improvement = (self.energy_efficiency / standard_vbf.energy_efficiency - 1) * 100
        print(f"{'Energy Efficiency (bits/μJ)':<30} {standard_vbf.energy_efficiency:<15.4f} {self.energy_efficiency:<20.4f} {efficiency_improvement:>9.1f}%")
        
        # Hop count
        hop_diff = self.hop_count - standard_vbf.hop_count
        hop_diff_percent = (hop_diff / max(1, standard_vbf.hop_count)) * 100
        print(f"{'Hop Count':<30} {standard_vbf.hop_count:<15d} {self.hop_count:<20d} {hop_diff_percent:>9.1f}%")
        
        # Path length
        std_path_length = standard_vbf._calculate_path_length()
        e_path_length = self._calculate_path_length()
        path_diff_percent = (e_path_length / std_path_length - 1) * 100
        print(f"{'Path Length (units)':<30} {std_path_length:<15.1f} {e_path_length:<20.1f} {path_diff_percent:>9.1f}%")
        
        # Delay
        delay_diff_percent = (self.total_delay / standard_vbf.total_delay - 1) * 100
        print(f"{'End-to-End Delay (ms)':<30} {standard_vbf.total_delay:<15.2f} {self.total_delay:<20.2f} {delay_diff_percent:>9.1f}%")
        
        # Throughput
        throughput_diff_percent = (self.throughput / standard_vbf.throughput - 1) * 100
        print(f"{'Throughput (bps)':<30} {standard_vbf.throughput:<15.2f} {self.throughput:<20.2f} {throughput_diff_percent:>9.1f}%")
        
        # PDR
        pdr_diff_percent = (self.packet_delivery_ratio / standard_vbf.packet_delivery_ratio - 1) * 100
        print(f"{'Packet Delivery Ratio':<30} {standard_vbf.packet_delivery_ratio:<15.4f} {self.packet_delivery_ratio:<20.4f} {pdr_diff_percent:>9.1f}%")
        
        print("-" * 80)
        print("\nEnergy Efficiency Techniques Used:")
        print("1. Adaptive Transmission Power: Adjust power based on distance")
        print("2. Energy-Aware Node Selection: Prefer nodes with higher energy")
        print("3. Duty Cycling: Only a subset of nodes active at any time")
        print("4. Energy Threshold: Exclude low-energy nodes from forwarding")
        print("===========================================\n")
        
        # Create a visualization of both paths
        plt.figure(figsize=(12, 10))
        self.visualize_static(compare_with_standard=True, standard_vbf_path=standard_vbf.path, 
                            save_path="vbf_comparison.png")

def simulate_and_compare():
    """Run simulation comparing standard VBF with energy-efficient VBF."""
    # Import the standard VBF implementation
    import vbf_simulation
    
    # Create standard VBF simulation
    print("Running Standard VBF Simulation...")
    standard_vbf = vbf_simulation.VBFSimulation(width=1000, height=1000, num_nodes=150, pipe_radius=120)
    standard_vbf.display_metrics_terminal()
    
    # Create energy-efficient VBF simulation with same parameters
    print("Running Energy-Efficient VBF Simulation...")
    e_vbf = EnergyEfficientVBF(width=1000, height=1000, num_nodes=150, pipe_radius=120)
    e_vbf.display_metrics_terminal()
    
    # Compare the two implementations
    e_vbf.compare_with_standard_vbf(standard_vbf)
    
    return standard_vbf, e_vbf

if __name__ == "__main__":
    standard_vbf, e_vbf = simulate_and_compare() 