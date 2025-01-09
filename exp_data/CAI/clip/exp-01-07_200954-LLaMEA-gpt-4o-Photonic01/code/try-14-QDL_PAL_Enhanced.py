import numpy as np

class QDL_PAL_Enhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.inertia_weight = 0.7  # Adjusted for better exploration
        self.cognitive_weight = 1.5  # Reduced to balance exploration and exploitation
        self.social_weight = 1.7  # Changed for stronger global influence
        self.sigma_min = 0.05  # Reduced for refined local exploration
        self.sigma_max = 0.3  # Reduced for stability in convergence
        self.iterations = budget // self.population_size
        self.adaptive_radius_min = 0.02  # Narrowed for precise local search
        self.adaptive_radius_max = 0.15  # Reduced to maintain balance

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        velocities = np.random.rand(self.population_size, self.dim) * 0.05 * (ub - lb)  # Reduced initial velocity
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

            r1, r2 = np.random.rand(2)
            velocities = (self.inertia_weight * velocities
                          + self.cognitive_weight * r1 * (personal_best_positions - population)
                          + self.social_weight * r2 * (global_best_position - population))
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

            adaptive_radius = self.adaptive_radius_min + (self.adaptive_radius_max - self.adaptive_radius_min) * (1 - progress_ratio)
            local_perturbation = np.random.uniform(-adaptive_radius, adaptive_radius, (self.population_size, self.dim))
            local_population = np.clip(new_population + local_perturbation, lb, ub)

            quantum_fitness = np.array([func(ind) for ind in quantum_population])
            local_fitness = np.array([func(ind) for ind in local_population])
            evals += 2 * self.population_size

            noise_selection_mask = np.random.rand(self.population_size) < 0.75  # Selective noise application
            improved_mask = (quantum_fitness < new_fitness) & (quantum_fitness < local_fitness) & noise_selection_mask
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