import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 30
        self.c1 = 2.0
        self.c2 = 2.0
        self.w = 0.5
        self.mutation_factor = 0.8
        self.crossover_prob = 0.9

    def __call__(self, func):
        np.random.seed(42)
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.population_size, np.inf)

        global_best_position = None
        global_best_score = np.inf

        evaluations = 0

        while evaluations < self.budget:
            # Evaluate the fitness of each particle
            fitness = np.apply_along_axis(func, 1, positions)
            
            # Update personal and global bests
            better_mask = fitness < personal_best_scores
            personal_best_scores[better_mask] = fitness[better_mask]
            personal_best_positions[better_mask] = positions[better_mask]

            min_index = np.argmin(personal_best_scores)
            if personal_best_scores[min_index] < global_best_score:
                global_best_score = personal_best_scores[min_index]
                global_best_position = personal_best_positions[min_index]

            # Update velocities and positions (PSO)
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (self.w * velocities 
                          + self.c1 * r1 * (personal_best_positions - positions)
                          + self.c2 * r2 * (global_best_position - positions)) * (0.5 + np.random.rand(self.population_size, self.dim)) # Enhanced

            positions = np.clip(positions + velocities, self.lower_bound, self.upper_bound)

            # Adaptive Differential Evolution (DE) Mutation
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = positions[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                cross_points = np.random.rand(self.dim) < self.crossover_prob
                trial = np.where(cross_points, mutant, positions[i])

                # Evaluate trial vector
                trial_fitness = func(trial)
                evaluations += 1

                # Selection
                if trial_fitness < personal_best_scores[i]:
                    personal_best_scores[i] = trial_fitness
                    personal_best_positions[i] = trial

            evaluations += self.population_size  # Add population size evaluations for PSO

        return global_best_position, global_best_score