import numpy as np

class HybridSwarmOptimizerACLOBLDIWSAMV2PFRefinedV3:
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
        self.cooling_rate = 0.99
        self.adaptive_cooling_rate = 0.5
        self.levy_flight_alpha = 1.5
        self.levy_flight_beta = 1.8
        self.opposition_based_learning_rate = 0.2
        self.dynamic_opposition_based_learning_rate = 0.1
        self.inertia_weight = 0.9
        self.inertia_weight_damping_ratio = 0.99
        self.mutation_rate = 0.1
        self.mutation_step_size = 0.1
        self.velocity_clustering_rate = 0.1
        self.particle_filtering_rate = 0.2
        self.archive_size = 10
        self.cauchy_mutation_rate = 0.05
        self.gaussian_perturbation_rate = 0.05

    def levy_flight(self, size):
        r1 = np.random.uniform(size=size)
        r2 = np.random.uniform(size=size)
        return 0.01 * r1 / (r2 ** (1 / self.levy_flight_beta))

    def cauchy_mutation(self, position):
        mutation_mask = np.random.rand(self.dim) < self.cauchy_mutation_rate
        mutation_vector = np.random.standard_cauchy(size=self.dim)
        return position + mutation_vector * mutation_mask

    def gaussian_perturbation(self, position):
        perturbation_vector = np.random.normal(loc=0, scale=0.1, size=self.dim)
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
                # Modified velocity update
                self.velocities[i] = self.inertia_weight * self.velocities[i] + 0.5 * np.random.uniform(-1, 1, size=self.dim) + 0.5 * (self.best_positions[i] - self.particles[i]) + 0.5 * (self.global_best_position - self.particles[i]) + 0.1 * np.random.uniform(-1, 1, size=self.dim)
                velocity_centroids = self.velocity_clustering(self.velocities)
                if np.random.rand() < self.velocity_clustering_rate:
                    self.velocities[i] = velocity_centroids[np.argmin(np.linalg.norm(self.velocities[i] - velocity_centroids, axis=1))]
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)
                self.velocities[i] = np.clip(self.velocities[i], -1, 1)
                # Levy flight for enhanced global search
                if np.random.rand() < 0.1:
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
                # Adaptive Cauchy mutation and Gaussian perturbation
                if np.random.rand() < self.cauchy_mutation_rate * (1 - evaluations / self.budget):
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
                if np.random.rand() < self.particle_filtering_rate:
                    particle_centroids = self.particle_filtering(self.particles)
                    self.particles[i] = particle_centroids[np.argmin(np.linalg.norm(self.particles[i] - particle_centroids, axis=1))]
                # Archive-based position update
                if np.random.rand() < 0.05:
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

optimizer = HybridSwarmOptimizerACLOBLDIWSAMV2PFRefinedV3(budget=1000, dim=10)
result = optimizer(example_func)
print(result)