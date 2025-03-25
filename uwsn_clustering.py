import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.distance import cdist

class UWSNClusteringProtocol:
    def __init__(self, n_nodes, world_size, init_energy, num_cluster_heads=3):
        """Initialize the UWSN Clustering Protocol
        
        Args:
            n_nodes (int): Number of nodes in the network
            world_size (float): Size of the world (square area)
            init_energy (float): Initial energy of each node
            num_cluster_heads (int): Number of cluster heads to select
        """
        self.n_nodes = n_nodes
        self.world_size = world_size
        self.init_energy = init_energy
        self.num_cluster_heads = num_cluster_heads  # Store the desired number of cluster heads
        
        # Initialize node positions randomly
        self.positions = np.random.uniform(0, world_size, size=(n_nodes, 2))
        
        # Initialize energy levels
        self.energy_levels = np.full(n_nodes, init_energy)
        
        # Place sink node at the center
        self.sink_position = np.array([world_size/2, world_size/2])
        
        # Initialize cluster-related attributes
        self.cluster_heads = []
        self.clusters = {}  # Maps cluster head ID to list of member node IDs
        self.transmission_range = world_size * 0.2  # Default 20% of world size
        
        # Performance metrics
        self.energy_consumption = []
        self.transmission_delay = []
        self.active_nodes = []
        self.dead_nodes = []
    
    def select_cluster_heads(self):
        """Select cluster heads based on energy levels and position"""
        # Reset cluster heads
        self.cluster_heads = []
        
        # Calculate scores for each node based on:
        # 1. Remaining energy (normalized)
        # 2. Distance to sink (normalized, inverse)
        # 3. Node density in neighborhood (normalized)
        
        energy_scores = self.energy_levels / self.init_energy
        
        # Calculate distances to sink
        distances_to_sink = np.linalg.norm(self.positions - self.sink_position, axis=1)
        max_distance = np.max(distances_to_sink)
        distance_scores = 1 - (distances_to_sink / max_distance)
        
        # Calculate node density scores
        density_scores = np.zeros(self.n_nodes)
        for i in range(self.n_nodes):
            distances = np.linalg.norm(self.positions - self.positions[i], axis=1)
            neighbors = np.sum(distances <= self.transmission_range) - 1  # Exclude self
            density_scores[i] = neighbors
        density_scores = density_scores / np.max(density_scores)
        
        # Combine scores (equal weights)
        total_scores = (energy_scores + distance_scores + density_scores) / 3
        
        # Select the top num_cluster_heads nodes as cluster heads
        candidate_indices = np.argsort(total_scores)[::-1]  # Sort in descending order
        
        # Filter out dead nodes and select cluster heads
        for idx in candidate_indices:
            if self.energy_levels[idx] > 0:  # Only select live nodes
                self.cluster_heads.append(idx)
                if len(self.cluster_heads) >= self.num_cluster_heads:
                    break
        
        # Clear previous cluster assignments
        self.clusters = {ch: [] for ch in self.cluster_heads}
    
    def form_clusters(self):
        """Form clusters by assigning nodes to nearest cluster head"""
        # Reset clusters
        self.clusters = {ch: [] for ch in self.cluster_heads}
        
        # Assign each node to nearest cluster head
        for node in range(self.n_nodes):
            if node not in self.cluster_heads and self.energy_levels[node] > 0:
                min_distance = float('inf')
                nearest_ch = None
                
                for ch in self.cluster_heads:
                    distance = np.linalg.norm(self.positions[node] - self.positions[ch])
                    if distance <= self.transmission_range and distance < min_distance:
                        min_distance = distance
                        nearest_ch = ch
                
                if nearest_ch is not None:
                    self.clusters[nearest_ch].append(node)
    
    def simulate_data_transfer(self):
        """Simulate one round of data transfer"""
        total_energy = 0
        total_delay = 0
        active_count = 0
        dead_count = 0
        
        # Energy consumption model parameters
        E_elec = 50e-9  # Energy for running transceiver circuit (J/bit)
        E_amp = 100e-12  # Energy for transmitter amplifier (J/bit/m^2)
        packet_size = 2000  # bits
        
        # Calculate energy consumption and delay for each cluster
        for ch in self.cluster_heads:
            if self.energy_levels[ch] <= 0:
                continue
                
            # Energy for receiving from cluster members
            for member in self.clusters[ch]:
                if self.energy_levels[member] <= 0:
                    continue
                    
                # Calculate distance between member and CH
                distance = np.linalg.norm(self.positions[member] - self.positions[ch])
                
                # Energy consumption for member node
                tx_energy = packet_size * (E_elec + E_amp * distance**2)
                self.energy_levels[member] -= tx_energy
                total_energy += tx_energy
                
                # Energy consumption for CH (receiving)
                rx_energy = packet_size * E_elec
                self.energy_levels[ch] -= rx_energy
                total_energy += rx_energy
                
                # Add transmission delay
                delay = distance / 1500  # Assuming speed of sound in water = 1500 m/s
                total_delay += delay
            
            # CH transmits aggregated data to sink
            distance_to_sink = np.linalg.norm(self.positions[ch] - self.sink_position)
            tx_energy_to_sink = packet_size * (E_elec + E_amp * distance_to_sink**2)
            self.energy_levels[ch] -= tx_energy_to_sink
            total_energy += tx_energy_to_sink
            
            # Add transmission delay to sink
            delay_to_sink = distance_to_sink / 1500
            total_delay += delay_to_sink
        
        # Update node status counts
        active_count = np.sum(self.energy_levels > 0)
        dead_count = self.n_nodes - active_count
        
        # Store metrics
        self.energy_consumption.append(total_energy)
        self.transmission_delay.append(total_delay)
        self.active_nodes.append(active_count)
        self.dead_nodes.append(dead_count)

    def plot_metrics(self):
        """Plot energy consumption, delay, and node status over time"""
        plt.figure(figsize=(18, 5))
        
        # Plot energy consumption
        plt.subplot(1, 3, 1)
        plt.plot(self.energy_consumption, label='Energy Consumption', color='blue')
        plt.xlabel('Time Frame')
        plt.ylabel('Energy (Joules)')
        plt.title('Energy Consumption Over Time')
        plt.grid(True)
        
        # Plot transmission delay
        plt.subplot(1, 3, 2)
        plt.plot(self.transmission_delay, label='Transmission Delay', color='red')
        plt.xlabel('Time Frame')
        plt.ylabel('Delay (ms)')
        plt.title('Transmission Delay Over Time')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

    def plot_node_status(self):
        """Plot active and dead nodes over time"""
        plt.figure(figsize=(9, 5))
        plt.plot(self.active_nodes, label='Active Nodes', color='green')
        plt.plot(self.dead_nodes, label='Dead Nodes', color='gray')
        plt.xlabel('Time Frame')
        plt.ylabel('Number of Nodes')
        plt.title('Active and Dead Nodes Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

    def visualize_network(self, frame=None):
        """Visualize the UWSN network with clusters in 2D and simulate data transfer"""
        plt.clf()
        
        # Plot regular nodes
        regular_nodes = np.array([i for i in range(self.n_nodes) if i not in self.cluster_heads])
        if len(regular_nodes) > 0:
            # Determine colors based on energy levels
            colors = ['gray' if self.energy_levels[node] <= 0 else 'blue' for node in regular_nodes]
            plt.scatter(self.positions[regular_nodes, 0],
                       self.positions[regular_nodes, 1],
                       c=colors, label='Sensor Nodes', alpha=0.6)
            
            # Annotate regular nodes with energy percentage
            for node in regular_nodes:
                energy_percentage = (self.energy_levels[node] / self.init_energy) * 100
                plt.annotate(f'{energy_percentage:.1f}%', (self.positions[node, 0], self.positions[node, 1]),
                             textcoords="offset points", xytext=(0,5), ha='center', fontsize=8, color='black')
        
        # Plot cluster heads
        plt.scatter(self.positions[self.cluster_heads, 0],
                   self.positions[self.cluster_heads, 1],
                   c='red', marker='^', s=100, label='Cluster Heads')
        
        # Annotate cluster heads with energy percentage
        for ch in self.cluster_heads:
            energy_percentage = (self.energy_levels[ch] / self.init_energy) * 100
            plt.annotate(f'{energy_percentage:.1f}%', (self.positions[ch, 0], self.positions[ch, 1]),
                         textcoords="offset points", xytext=(0,5), ha='center', fontsize=8, color='black')
        
        # Plot sink node
        plt.scatter(self.sink_position[0], self.sink_position[1], c='purple', marker='s', s=150, label='Sink')
        plt.annotate("SINK", (self.sink_position[0], self.sink_position[1]), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=12, 
                    color='purple', weight='bold')
        
        # Draw cluster boundaries as shaded regions
        for ch in self.cluster_heads:
            if len(self.clusters[ch]) > 0:
                # Create a convex hull around cluster members
                member_positions = np.vstack([self.positions[ch], 
                                            np.array([self.positions[m] for m in self.clusters[ch] 
                                                    if self.energy_levels[m] > 0])])
                if len(member_positions) > 3:  # Need at least 3 points for convex hull
                    try:
                        from scipy.spatial import ConvexHull
                        hull = ConvexHull(member_positions)
                        # Plot the convex hull
                        for simplex in hull.simplices:
                            plt.plot(member_positions[simplex, 0], member_positions[simplex, 1], 
                                   'r-', alpha=0.2)
                        # Fill the convex hull
                        hull_points = member_positions[hull.vertices]
                        plt.fill(hull_points[:, 0], hull_points[:, 1], 
                               alpha=0.1, color='red')
                    except:
                        # Fallback if convex hull fails
                        pass
        
        # Draw lines between cluster heads and their member nodes
        for ch, members in self.clusters.items():
            for member in members:
                if self.energy_levels[member] > 0:  # Only draw lines for active nodes
                    plt.plot([self.positions[ch, 0], self.positions[member, 0]],
                            [self.positions[ch, 1], self.positions[member, 1]],
                            'g-', alpha=0.2)
        
        # Visualize data transfer phases
        if frame is not None:
            # Phase 1: Data collection from members to cluster heads (frames 1-3)
            if frame >= 1 and frame <= 3:
                phase = frame  # 1, 2, or 3 representing progress within phase 1
                
                for ch, members in self.clusters.items():
                    if self.energy_levels[ch] <= 0:
                        continue
                        
                    active_members = [m for m in members if self.energy_levels[m] > 0]
                    # Only show a subset of members per frame to create animation effect
                    members_to_show = active_members[:int(len(active_members) * phase/3)]
                    
                    for member in members_to_show:
                        # Draw data packets moving from members to CH
                        vector = self.positions[ch] - self.positions[member]
                        packet_pos = self.positions[member] + (phase/3) * vector
                        
                        # Show the data packet
                        plt.scatter(packet_pos[0], packet_pos[1], 
                                  c='cyan', marker='o', s=80, edgecolor='blue', 
                                  alpha=0.8, zorder=10)
                        
                        # Show the data path
                        plt.arrow(self.positions[member, 0], self.positions[member, 1],
                                vector[0] * phase/3, vector[1] * phase/3,
                                head_width=10, head_length=15, fc='blue', ec='blue',
                                alpha=0.5, zorder=4, length_includes_head=True)
            
            # Phase 2: Data aggregation at cluster heads (frames 4-5)
            elif frame >= 4 and frame <= 5:
                # Show aggregation effect at cluster heads
                for ch in self.cluster_heads:
                    if self.energy_levels[ch] <= 0 or len(self.clusters[ch]) == 0:
                        continue
                        
                    # Create pulsating effect around cluster head
                    circle_size = 30 + 10 * (frame - 3)  # Size increases with frame
                    agg_circle = plt.Circle((self.positions[ch, 0], self.positions[ch, 1]), 
                                         circle_size, color='cyan', alpha=0.3, zorder=3)
                    plt.gca().add_patch(agg_circle)
            
            # Phase 3: Data transmission to sink (frames 6-10)
            elif frame >= 6:
                # Normalize for progress within phase 3
                phase_progress = (frame - 5) / 5  # 0.2, 0.4, 0.6, 0.8, 1.0
                
                for ch in self.cluster_heads:
                    if self.energy_levels[ch] <= 0 or len(self.clusters[ch]) == 0:
                        continue
                    
                    # Draw thicker arrow line from cluster head to sink
                    plt.arrow(self.positions[ch, 0], self.positions[ch, 1],
                            (self.sink_position[0] - self.positions[ch, 0]) * 0.9,
                            (self.sink_position[1] - self.positions[ch, 1]) * 0.9,
                            head_width=15, head_length=20, fc='orange', ec='orange',
                            alpha=0.7, zorder=5, length_includes_head=True)
                    
                    # Draw data packet moving along the path
                    vector = self.sink_position - self.positions[ch]
                    packet_pos = self.positions[ch] + phase_progress * vector
                    
                    # Show the data packet
                    plt.scatter(packet_pos[0], packet_pos[1], 
                              c='orange', marker='o', s=100, edgecolor='red', 
                              alpha=1.0, zorder=10)
                
                # If frame is >= 9, show reception at sink node
                if frame >= 9:
                    # Add a glow effect around the sink
                    sink_glow = plt.Circle((self.sink_position[0], self.sink_position[1]), 
                                       70, color='yellow', alpha=0.2, zorder=2)
                    plt.gca().add_patch(sink_glow)
        
        # Draw transmission range circles for cluster heads
        for ch in self.cluster_heads:
            circle = plt.Circle((self.positions[ch, 0], self.positions[ch, 1]), 
                              self.transmission_range, 
                              fill=False, 
                              linestyle='--', 
                              color='gray', 
                              alpha=0.3)
            plt.gca().add_patch(circle)
        
        # Add network statistics on the plot
        active_count = np.sum(self.energy_levels > 0)
        plt.text(0.02, 0.02, f"Active Nodes: {active_count}/{self.n_nodes}", transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8), fontsize=10)
        plt.text(0.02, 0.06, f"Cluster Heads: {len(self.cluster_heads)}", transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8), fontsize=10)
        
        # Add data flow phase information
        phase_text = "Setup"
        if frame is not None:
            if 1 <= frame <= 3:
                phase_text = "Phase 1: Data Collection"
            elif 4 <= frame <= 5:
                phase_text = "Phase 2: Data Aggregation"
            elif 6 <= frame <= 10:
                phase_text = "Phase 3: Transmission to Sink"
                
        plt.text(0.02, 0.10, f"Frame: {frame if frame is not None else 0} - {phase_text}", 
                transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8), fontsize=10)
        
        # Add packet count information
        total_packets = sum(len(members) for members in self.clusters.values())
        plt.text(0.02, 0.14, f"Total Packets: {total_packets}", transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.8), fontsize=10)
                
        # Simulate data transfer
        self.simulate_data_transfer()
        
        plt.xlabel('X Distance (m)')
        plt.ylabel('Y Distance (m)')
        plt.title('UWSN Adaptive Clustering Protocol Simulation')
        
        # Add legend with correct colors
        handles = [plt.Line2D([0], [0], marker='o', color='w', label='Sensor Nodes', markersize=10, markerfacecolor='blue'),
                   plt.Line2D([0], [0], marker='^', color='w', label='Cluster Heads', markersize=10, markerfacecolor='red'),
                   plt.Line2D([0], [0], marker='o', color='w', label='Dead Nodes', markersize=10, markerfacecolor='gray'),
                   plt.Line2D([0], [0], marker='s', color='w', label='Sink', markersize=10, markerfacecolor='purple'),
                   plt.Line2D([0], [0], marker='o', color='w', label='CH→Sink Packet', markersize=10, markerfacecolor='orange', markeredgecolor='red'),
                   plt.Line2D([0], [0], marker='o', color='w', label='Node→CH Packet', markersize=10, markerfacecolor='cyan', markeredgecolor='blue')]
        plt.legend(handles=handles, loc='upper right')
        
        plt.grid(True, alpha=0.3)
        plt.axis('equal')  # Make the plot aspect ratio 1:1
        plt.xlim(0, self.world_size)
        plt.ylim(0, self.world_size)

def main():
    # Initialize the UWSN protocol
    uwsn = UWSNClusteringProtocol(n_nodes=50, world_size=1000, init_energy=100)
    
    # Create figure for animation
    plt.figure(figsize=(12, 10))
    
    def update(frame):
        # Only select cluster heads in first frame or if we have none
        if frame == 0 or not uwsn.cluster_heads:
            uwsn.select_cluster_heads()
            uwsn.form_clusters()
        
        uwsn.visualize_network(frame)
        
        # Print parameters to the terminal
        print(f"Frame {frame + 1}:")
        print(f"Number of Nodes: {uwsn.n_nodes}")
        print(f"World Size: {uwsn.world_size}")
        print(f"Initial Energy: {uwsn.init_energy}")
        print(f"Number of Cluster Heads: {len(uwsn.cluster_heads)}")
        print(f"Cluster Head IDs: {uwsn.cluster_heads}")
        
        # Print cluster information
        total_members = 0
        for ch, members in uwsn.clusters.items():
            print(f"Cluster Head {ch}: {len(members)} members")
            total_members += len(members)
        
        print(f"Total Cluster Members: {total_members}")
        print(f"Active Nodes: {np.sum(uwsn.energy_levels > 0)}/{uwsn.n_nodes}")
        print("\n")
    
    # Create animation with more frames to show data transfer
    anim = FuncAnimation(plt.gcf(), update, frames=10, interval=1500)
    
    # Save animation
    try:
        print("Saving animation as 'uwsn_clustering.gif'...")
        anim.save('uwsn_clustering.gif', writer='pillow', fps=1, dpi=120)
        print("Animation saved successfully!")
    except Exception as e:
        print(f"Failed to save animation: {e}")
        
    plt.show()
    
    # Plot energy consumption and delay
    uwsn.plot_metrics()
    
    # Plot active and dead nodes
    uwsn.plot_node_status()

    # Calculate and display average metrics
    avg_energy_consumption = np.mean(uwsn.energy_consumption)
    avg_delay = np.mean(uwsn.transmission_delay)
    total_packets_transferred = 0
    for ch, members in uwsn.clusters.items():
        total_packets_transferred += len(members)
    
    print("Average Energy Consumption (Joules):", avg_energy_consumption)
    print("Average Transmission Delay (ms):", avg_delay)
    print("Total Packets Transferred:", total_packets_transferred)

if __name__ == "__main__":
    main() 