import numpy as np

class QDL_PAL_Enhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.inertia_weight = 0.5  # Adjusted for enhanced exploration
        self.cognitive_weight = 1.8  # Enhanced personal search
        self.social_weight = 1.4  # Slight increase for better convergence
        self.sigma_min = 0.05  # Reduced for finer local search
        self.sigma_max = 0.35  # Smoothed convergence
        self.iterations = budget // self.population_size
        self.adaptive_radius = 0.12  # Slight increase for local search
        self.communication_factor = 0.2  # New parameter for inter-agent collaboration

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        velocities = np.random.rand(self.population_size, self.dim) * 0.1 * (ub - lb)
        fitness = np.array([func(ind) for ind in population])
        evals = self.population_size

        personal_best_positions = np.copy(population)
        personal_best_fitness = np.copy(fitness)

        global_best_idx = np.argmin(fitness)
        global_best_position = population[global_best_idx]
        global_best_fitness = fitness[global_best_idx]

        for iteration in range(self.iterations):
            if evals >= self.budget:
                break

            r1, r2, r3 = np.random.rand(3)
            velocities = (self.inertia_weight * velocities
                          + self.cognitive_weight * r1 * (personal_best_positions - population)
                          + self.social_weight * r2 * (global_best_position - population)
                          - self.communication_factor * r3 * (population.mean(axis=0) - population))
            new_population = np.clip(population + velocities, lb, ub)

            new_fitness = np.array([func(ind) for ind in new_population])
            evals += self.population_size

            better_mask = new_fitness < personal_best_fitness
            personal_best_positions[better_mask] = new_population[better_mask]
            personal_best_fitness[better_mask] = new_fitness[better_mask]

            min_new_fitness_idx = np.argmin(new_fitness)
            if new_fitness[min_new_fitness_idx] < global_best_fitness:
                global_best_position = new_population[min_new_fitness_idx]
                global_best_fitness = new_fitness[min_new_fitness_idx]

            progress_ratio = iteration / self.iterations
            sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * progress_ratio

            gaussian_noise = np.random.normal(0, sigma, (self.population_size, self.dim))
            quantum_population = np.clip(new_population + gaussian_noise, lb, ub)

            local_perturbation = np.random.uniform(-self.adaptive_radius, self.adaptive_radius,
                                                   (self.population_size, self.dim))
            local_population = np.clip(new_population + local_perturbation, lb, ub)

            quantum_fitness = np.array([func(ind) for ind in quantum_population])
            local_fitness = np.array([func(ind) for ind in local_population])
            evals += 2 * self.population_size

            improved_mask = (quantum_fitness < new_fitness) & (quantum_fitness < local_fitness)
            population[improved_mask] = quantum_population[improved_mask]
            fitness[improved_mask] = quantum_fitness[improved_mask]

            local_improved_mask = local_fitness < fitness
            population[local_improved_mask] = local_population[local_improved_mask]
            fitness[local_improved_mask] = local_fitness[local_improved_mask]

            if np.min(fitness) < global_best_fitness:
                global_best_idx = np.argmin(fitness)
                global_best_position = population[global_best_idx]
                global_best_fitness = fitness[global_best_idx]

        return global_best_position, global_best_fitness