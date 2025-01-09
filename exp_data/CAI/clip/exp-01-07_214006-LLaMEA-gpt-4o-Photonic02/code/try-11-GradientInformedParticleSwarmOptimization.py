import numpy as np

class GradientInformedParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.current_evaluations = 0
        self.w = 0.5  # Inertia weight
        self.c1 = 2.0  # Cognitive component
        self.c2 = 2.0  # Social component
        self.gradient_step_size = 0.01

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        velocities = np.random.rand(self.population_size, self.dim) * (bounds[:, 1] - bounds[:, 0]) / 20
        personal_best_positions = np.copy(population)
        personal_best_fitness = np.array([func(ind) for ind in population])
        global_best_index = np.argmin(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_index]
        self.current_evaluations += self.population_size

        while self.current_evaluations < self.budget:
            for i in range(self.population_size):
                # Velocity update
                velocities[i] = (self.w * velocities[i] +
                                self.c1 * np.random.rand(self.dim) * (personal_best_positions[i] - population[i]) +
                                self.c2 * np.random.rand(self.dim) * (global_best_position - population[i]))
                # Position update
                population[i] = population[i] + velocities[i]
                population[i] = np.clip(population[i], bounds[:, 0], bounds[:, 1])
                
                # Gradient-informed search adjustment
                gradient_estimate = self.estimate_gradient(func, population[i], bounds)
                population[i] = population[i] - self.gradient_step_size * gradient_estimate
                population[i] = np.clip(population[i], bounds[:, 0], bounds[:, 1])
                
                # Evaluate and update personal and global bests
                current_fitness = func(population[i])
                self.current_evaluations += 1

                if current_fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = population[i]
                    personal_best_fitness[i] = current_fitness
                    if current_fitness < personal_best_fitness[global_best_index]:
                        global_best_position = population[i]
                        global_best_index = i

            if self.current_evaluations >= self.budget:
                break

        return global_best_position, personal_best_fitness[global_best_index]

    def estimate_gradient(self, func, position, bounds):
        # Estimate gradient via finite differences
        epsilon = 1e-8
        gradient = np.zeros(self.dim)
        for d in range(self.dim):
            pos = np.copy(position)
            pos[d] += epsilon
            pos = np.clip(pos, bounds[d, 0], bounds[d, 1])
            gradient[d] = (func(pos) - func(position)) / epsilon
            self.current_evaluations += 1
            if self.current_evaluations >= self.budget:
                break
        return gradient