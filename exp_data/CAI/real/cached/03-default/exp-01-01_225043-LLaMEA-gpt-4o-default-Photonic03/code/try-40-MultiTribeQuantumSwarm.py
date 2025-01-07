import numpy as np

class MultiTribeQuantumSwarm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.num_tribes = 5
        self.tribe_size = self.population_size // self.num_tribes
        self.particles = np.random.uniform(size=(self.population_size, dim))
        self.velocities = np.random.uniform(size=(self.population_size, dim)) * 0.1
        self.tribal_best = self.particles[:self.num_tribes].copy()
        self.global_best = None
        self.tribal_best_fitness = np.full(self.num_tribes, np.inf)
        self.global_best_fitness = np.inf
        self.fitness_evaluations = 0
        self.archive = []

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        lower_bound, upper_bound = bounds[0], bounds[1]

        quantum_jump_prob = lambda evals: 0.3 - 0.1 * (evals / self.budget)

        while self.fitness_evaluations < self.budget:
            for tribe in range(self.num_tribes):
                for i in range(tribe * self.tribe_size, (tribe + 1) * self.tribe_size):
                    if self.fitness_evaluations >= self.budget:
                        break

                    fitness = func(self.particles[i])
                    self.fitness_evaluations += 1

                    if fitness < self.tribal_best_fitness[tribe]:
                        self.tribal_best_fitness[tribe] = fitness
                        self.tribal_best[tribe] = self.particles[i].copy()

                    if fitness < self.global_best_fitness:
                        self.global_best_fitness = fitness
                        self.global_best = self.particles[i].copy()

            for tribe in range(self.num_tribes):
                if self.fitness_evaluations >= self.budget:
                    break

                for i in range(tribe * self.tribe_size, (tribe + 1) * self.tribe_size):
                    inertia_weight = 0.7 - 0.4 * (self.fitness_evaluations / self.budget)
                    cognitive_coeff = 1.4
                    social_coeff = 1.7
                    r1, r2 = np.random.rand(), np.random.rand()

                    cognitive_velocity = cognitive_coeff * r1 * (self.tribal_best[tribe] - self.particles[i])
                    social_velocity = social_coeff * r2 * (self.global_best - self.particles[i])
                    self.velocities[i] = inertia_weight * self.velocities[i] + cognitive_velocity + social_velocity
                    self.particles[i] += self.velocities[i]
                    self.particles[i] = np.clip(self.particles[i], lower_bound, upper_bound)

                    if np.random.rand() < quantum_jump_prob(self.fitness_evaluations):
                        quantum_exploration = lower_bound + np.random.rand(self.dim) * (upper_bound - lower_bound)
                        self.particles[i] = quantum_exploration

            if self.fitness_evaluations % (self.budget // 10) == 0:
                best_tribe_idx = np.argmin(self.tribal_best_fitness)
                self.particles = self.reinitialize_tribes(best_tribe_idx, lower_bound, upper_bound)

        return self.global_best

    def reinitialize_tribes(self, best_tribe_idx, lower_bound, upper_bound):
        new_particles = np.random.uniform(size=(self.population_size, self.dim))
        new_particles[best_tribe_idx * self.tribe_size:(best_tribe_idx + 1) * self.tribe_size] = self.particles[best_tribe_idx * self.tribe_size:(best_tribe_idx + 1) * self.tribe_size]
        return new_particles