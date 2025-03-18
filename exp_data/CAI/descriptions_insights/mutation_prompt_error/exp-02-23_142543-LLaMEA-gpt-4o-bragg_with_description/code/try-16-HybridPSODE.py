import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.c1 = 2.0  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient
        self.w = 0.5   # Inertia weight (reduced for better exploration-exploitation balance)
        self.F = 0.6   # DE scale factor (increased for stronger mutation influence)
        self.CR = 0.9  # DE crossover probability

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.random.uniform(-abs(ub-lb), abs(ub-lb), (self.population_size, self.dim))
        personal_best = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in personal_best])
        global_best_idx = np.argmin(personal_best_scores)
        global_best = personal_best[global_best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            # Update velocity and population (PSO)
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocity = (self.w * velocity
                        + self.c1 * r1 * (personal_best - population)
                        + self.c2 * r2 * (global_best - population))
            population = np.clip(population + velocity, lb, ub)

            # Evaluate fitness
            fitness = np.array([func(ind) for ind in population])
            evaluations += self.population_size

            # Update personal bests
            better_fit = fitness < personal_best_scores
            personal_best[better_fit] = population[better_fit]
            personal_best_scores[better_fit] = fitness[better_fit]

            # Update global best
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < personal_best_scores[global_best_idx]:
                global_best_idx = current_best_idx
                global_best = population[global_best_idx]

            # Differential Evolution Mutation
            for i in range(self.population_size):
                adaptive_CR = self.CR - (self.CR * evaluations / self.budget)
                if np.random.rand() < adaptive_CR:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                    x0, x1, x2 = population[indices]
                    mutant = np.clip(x0 + self.F * (x1 - x2), lb, ub)

                    # Refined Periodicity-based Mutation
                    if evaluations < self.budget:
                        period = (ub - lb) / (self.dim * 1.5)
                        mutant = lb + np.abs((mutant - lb) % (2 * period) - period)

                        # Evaluate mutant
                        mutant_fitness = func(mutant) + 1e-5 * np.linalg.norm(mutant - global_best)  # Modified Line 1
                        evaluations += 1

                        # Selection
                        if mutant_fitness < fitness[i]:
                            population[i] = mutant
                            fitness[i] = mutant_fitness
                            if mutant_fitness < personal_best_scores[i]:
                                personal_best[i] = mutant
                                personal_best_scores[i] = mutant_fitness
                                if mutant_fitness < personal_best_scores[global_best_idx]:
                                    global_best_idx = i
                                    global_best = mutant

        return global_best