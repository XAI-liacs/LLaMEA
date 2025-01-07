import numpy as np

class AdaptiveQuantumGeneticPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, dim)
        self.inertia = 0.7
        self.c1 = 1.5
        self.c2 = 1.5
        self.crossover_rate = 0.8
        self.mutation_rate = 0.2
        self.beta = 0.05

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim)) * (ub - lb)
        personal_best_positions = population.copy()
        personal_best_scores = np.array([func(population[i]) for i in range(self.population_size)])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = population[global_best_index].copy()
        evaluations = self.population_size

        while evaluations < self.budget:
            # Adaptive mutation and crossover rates
            self.crossover_rate = 0.6 + 0.4 * (1 - evaluations / self.budget)
            self.mutation_rate = 0.1 + 0.3 * (evaluations / self.budget)

            # Quantum-inspired genetic operations
            new_population = np.empty((self.population_size, self.dim))
            for i in range(0, self.population_size, 2):
                parent1, parent2 = population[np.random.choice(self.population_size, 2, replace=False)]
                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                new_population[i] = np.where(crossover_mask, parent1, parent2)
                if i + 1 < self.population_size:
                    new_population[i + 1] = np.where(crossover_mask, parent2, parent1)

                mutation_mask = np.random.rand(self.dim) < self.mutation_rate
                new_population[i] += mutation_mask * np.random.uniform(-0.1, 0.1, self.dim) * (ub - lb)
                if i + 1 < self.population_size:
                    new_population[i + 1] += mutation_mask * np.random.uniform(-0.1, 0.1, self.dim) * (ub - lb)
                
                # Quantum-inspired adjustment
                if np.random.rand() < self.beta:
                    q = np.random.normal(loc=0, scale=1)
                    new_population[i] = global_best_position + q * (ub - lb)
                    if i + 1 < self.population_size:
                        new_population[i + 1] = global_best_position + q * (ub - lb)

            # PSO velocity and position update
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (self.inertia * velocities +
                          self.c1 * r1 * (personal_best_positions - population) +
                          self.c2 * r2 * (global_best_position - population))
            population = np.clip(population + velocities, lb, ub)

            # Evaluate new population
            new_scores = np.array([func(new_population[i]) for i in range(self.population_size)])
            evaluations += self.population_size

            # Update personal and global bests
            update_mask = new_scores < personal_best_scores
            personal_best_positions[update_mask] = new_population[update_mask]
            personal_best_scores[update_mask] = new_scores[update_mask]

            new_global_best_index = np.argmin(personal_best_scores)
            if personal_best_scores[new_global_best_index] < personal_best_scores[global_best_index]:
                global_best_index = new_global_best_index
                global_best_position = personal_best_positions[global_best_index]

        # Return the best solution found
        return global_best_position, personal_best_scores[global_best_index]