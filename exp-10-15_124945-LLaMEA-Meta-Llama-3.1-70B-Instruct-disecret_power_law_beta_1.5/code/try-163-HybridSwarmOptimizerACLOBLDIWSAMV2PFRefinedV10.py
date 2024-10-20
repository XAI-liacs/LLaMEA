import numpy as np

class HybridSwarmOptimizerACLOBLDIWSAMV2PFRefinedV10:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.swarm_size = int(np.sqrt(budget))
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.swarm_size, self.dim))
        self.velocities = np.zeros((self.swarm_size, self.dim))
        self.best_positions = np.copy(self.particles)
        self.best_fitness = np.inf * np.ones(self.swarm_size)
        self.global_best_position = np.random.uniform(self.lower_bound, self.upper_bound, size=self.dim)
        self.global_best_fitness = np.inf
        self.archive = []
        self.temperature = 1000.0
        self.cooling_rate = 0.98  # Changed from 0.99 to 0.98
        self.adaptive_cooling_rate = 0.3  # Changed from 0.4 to 0.3
        self.levy_flight_alpha = 1.2  # Changed from 1.5 to 1.2
        self.levy_flight_beta = 1.5  # Changed from 1.8 to 1.5
        self.opposition_based_learning_rate = 0.25  # Changed from 0.2 to 0.25
        self.dynamic_opposition_based_learning_rate = 0.1  # Changed from 0.15 to 0.1
        self.inertia_weight = 0.85  # Changed from 0.9 to 0.85
        self.inertia_weight_damping_ratio = 0.995  # Changed from 0.99 to 0.995
        self.mutation_rate = 0.12  # Changed from 0.1 to 0.12
        self.mutation_step_size = 0.12  # Changed from 0.1 to 0.12
        self.velocity_clustering_rate = 0.12  # Changed from 0.1 to 0.12
        self.particle_filtering_rate = 0.22  # Changed from 0.2 to 0.22
        self.adaptive_particle_filtering_rate = 0.055  # Changed from 0.05 to 0.055
        self.archive_size = 12  # Changed from 10 to 12
        self.exploration_rate = 0.55  # Changed from 0.5 to 0.55
        self.exploitation_rate = 0.45  # Changed from 0.5 to 0.45
        self.hybrid_repulsion_rate = 0.12  # Changed from 0.1 to 0.12
        self.swarm_restructuring_rate = 0.055  # Changed from 0.05 to 0.055
        self.quantum_tunneling_rate = 0.12  # Changed from 0.1 to 0.12

    def levy_flight(self, size):
        r1 = np.random.uniform(size=size)
        r2 = np.random.uniform(size=size)
        return 0.01 * r1 / (r2 ** (1 / self.levy_flight_beta))

    def cauchy_mutation(self, position):
        mutation_mask = np.random.rand(self.dim) < self.mutation_rate
        mutation_vector = np.random.standard_cauchy(size=self.dim)
        return position + mutation_vector * mutation_mask

    def gaussian_perturbation(self, position):
        perturbation_vector = np.random.normal(loc=0, scale=0.12, size=self.dim)  # Changed from 0.1 to 0.12
        return position + perturbation_vector

    def opposition_based_learning(self, position):
        return self.lower_bound + self.upper_bound - position

    def velocity_clustering(self, velocities):
        velocity_centroids = np.random.uniform(-1, 1, size=(3, self.dim))
        velocity_clusters = np.argmin(np.linalg.norm(velocities[:, np.newaxis] - velocity_centroids, axis=2), axis=1)
        for i in range(3):
            cluster_velocities = velocities[velocity_clusters == i]
            if len(cluster_velocities) > 0:
                velocity_centroids[i] = np.mean(cluster_velocities, axis=0)
        return velocity_centroids

    def particle_filtering(self, particles):
        particle_centroids = np.random.uniform(-1, 1, size=(3, self.dim))
        particle_clusters = np.argmin(np.linalg.norm(particles[:, np.newaxis] - particle_centroids, axis=2), axis=1)
        for i in range(3):
            cluster_particles = particles[particle_clusters == i]
            if len(cluster_particles) > 0:
                particle_centroids[i] = np.mean(cluster_particles, axis=0)
        return particle_centroids

    def hybrid_repulsion(self, position):
        repulsion_vector = np.random.uniform(-1, 1, size=self.dim)
        return position + repulsion_vector * np.random.uniform(0, 1, size=self.dim)

    def swarm_restructuring(self, particles):
        centroid = np.mean(particles, axis=0)
        distances = np.linalg.norm(particles - centroid, axis=1)
        farthest_index = np.argmax(distances)
        return particles[farthest_index]

    def quantum_tunneling(self, position):
        tunneling_vector = np.random.uniform(-1, 1, size=self.dim)
        return position + tunneling_vector * np.random.uniform(0, 1, size=self.dim)

    def update_archive(self, position, fitness):
        if len(self.archive) < self.archive_size:
            self.archive.append((position, fitness))
        else:
            worst_index = np.argmax([f for _, f in self.archive])
            if fitness < self.archive[worst_index][1]:
                self.archive[worst_index] = (position, fitness)

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.swarm_size):
                fitness = func(self.particles[i])
                evaluations += 1
                if fitness < self.best_fitness[i]:
                    self.best_fitness[i] = fitness
                    self.best_positions[i] = np.copy(self.particles[i])
                    self.update_archive(self.particles[i], fitness)
                    if fitness < self.global_best_fitness:
                        self.global_best_fitness = fitness
                        self.global_best_position = np.copy(self.particles[i])
                # Modified velocity update with adaptive acceleration coefficients
                self.velocities[i] = self.inertia_weight * self.velocities[i] + 0.5 * np.random.uniform(-1, 1, size=self.dim) * (1 - evaluations / self.budget) + 0.5 * (self.best_positions[i] - self.particles[i]) * (1 + evaluations / self.budget) + 0.5 * (self.global_best_position - self.particles[i]) * (1 - evaluations / self.budget) + 0.12 * np.random.uniform(-1, 1, size=self.dim)  # Changed from 0.1 to 0.12
                velocity_centroids = self.velocity_clustering(self.velocities)
                if np.random.rand() < self.velocity_clustering_rate:
                    self.velocities[i] = velocity_centroids[np.argmin(np.linalg.norm(self.velocities[i] - velocity_centroids, axis=1))]
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)
                self.velocities[i] = np.clip(self.velocities[i], -1, 1)
                # Levy flight for enhanced global search
                if np.random.rand() < 0.12:  # Changed from 0.1 to 0.12
                    self.particles[i] += self.levy_flight(self.dim)
                    self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)
                # Dynamic opposition-based learning with adaptive rate
                if np.random.rand() < self.opposition_based_learning_rate * (1 - evaluations / self.budget) * (1 + self.dynamic_opposition_based_learning_rate * (evaluations / self.budget)):
                    opposition_position = self.opposition_based_learning(self.particles[i])
                    opposition_fitness = func(opposition_position)
                    evaluations += 1
                    if opposition_fitness < fitness:
                        self.particles[i] = opposition_position
                        self.best_fitness[i] = opposition_fitness
                        self.best_positions[i] = np.copy(self.particles[i])
                        self.update_archive(self.particles[i], opposition_fitness)
                # Cauchy mutation and Gaussian perturbation
                if np.random.rand() < self.mutation_rate * (1 - evaluations / self.budget):
                    mutated_position = self.cauchy_mutation(self.particles[i])
                    mutated_position = self.gaussian_perturbation(mutated_position)
                    mutated_fitness = func(mutated_position)
                    evaluations += 1
                    if mutated_fitness < fitness:
                        self.particles[i] = mutated_position
                        self.best_fitness[i] = mutated_fitness
                        self.best_positions[i] = np.copy(self.particles[i])
                        self.update_archive(self.particles[i], mutated_fitness)
                # Particle filtering for enhanced exploration
                if np.random.rand() < self.particle_filtering_rate * (1 - evaluations / self.budget) * (1 + self.adaptive_particle_filtering_rate * (evaluations / self.budget)):
                    particle_centroids = self.particle_filtering(self.particles)
                    self.particles[i] = particle_centroids[np.argmin(np.linalg.norm(self.particles[i] - particle_centroids, axis=1))]
                # Hybrid-repulsion mechanism
                if np.random.rand() < self.hybrid_repulsion_rate * (1 - evaluations / self.budget):
                    repulsion_position = self.hybrid_repulsion(self.particles[i])
                    repulsion_fitness = func(repulsion_position)
                    evaluations += 1
                    if repulsion_fitness < fitness:
                        self.particles[i] = repulsion_position
                        self.best_fitness[i] = repulsion_fitness
                        self.best_positions[i] = np.copy(self.particles[i])
                        self.update_archive(self.particles[i], repulsion_fitness)
                # Swarm restructuring mechanism
                if np.random.rand() < self.swarm_restructuring_rate * (1 - evaluations / self.budget):
                    restructuring_position = self.swarm_restructuring(self.particles)
                    restructuring_fitness = func(restructuring_position)
                    evaluations += 1
                    if restructuring_fitness < fitness:
                        self.particles[i] = restructuring_position
                        self.best_fitness[i] = restructuring_fitness
                        self.best_positions[i] = np.copy(self.particles[i])
                        self.update_archive(self.particles[i], restructuring_fitness)
                # Quantum tunneling for enhanced local search
                if np.random.rand() < self.quantum_tunneling_rate * (1 - evaluations / self.budget):
                    tunneling_position = self.quantum_tunneling(self.particles[i])
                    tunneling_fitness = func(tunneling_position)
                    evaluations += 1
                    if tunneling_fitness < fitness:
                        self.particles[i] = tunneling_position
                        self.best_fitness[i] = tunneling_fitness
                        self.best_positions[i] = np.copy(self.particles[i])
                        self.update_archive(self.particles[i], tunneling_fitness)
                # Exploration-exploitation balance mechanism
                if np.random.rand() < self.exploration_rate:
                    self.particles[i] += np.random.uniform(-1, 1, size=self.dim)
                    self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)
                else:
                    self.particles[i] += 0.5 * (self.best_positions[i] - self.particles[i]) + 0.5 * (self.global_best_position - self.particles[i])
                    self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)
                # Archive-based position update
                if np.random.rand() < 0.055:  # Changed from 0.05 to 0.055
                    archive_index = np.random.randint(len(self.archive))
                    self.particles[i] = self.archive[archive_index][0]
            # Modified simulated annealing with adaptive cooling
            if evaluations % (self.swarm_size // 2) == 0:
                for i in range(self.swarm_size // 2):
                    new_position = np.copy(self.particles[i])
                    new_position += np.random.uniform(-1, 1, size=self.dim)
                    new_position = np.clip(new_position, self.lower_bound, self.upper_bound)
                    new_fitness = func(new_position)
                    evaluations += 1
                    if new_fitness < self.best_fitness[i] or np.random.rand() < np.exp((self.best_fitness[i] - new_fitness) / self.temperature):
                        self.particles[i] = new_position
                        self.best_fitness[i] = new_fitness
                        self.best_positions[i] = np.copy(self.particles[i])
                        self.update_archive(self.particles[i], new_fitness)
                    self.temperature *= self.cooling_rate * (1 - self.adaptive_cooling_rate * (evaluations / self.budget))
            # Dynamic inertia weight
            self.inertia_weight *= self.inertia_weight_damping_ratio
        return self.global_best_position

# Example usage:
def example_func(x):
    return np.sum(x**2)

optimizer = HybridSwarmOptimizerACLOBLDIWSAMV2PFRefinedV10(budget=1000, dim=10)
result = optimizer(example_func)
print(result)