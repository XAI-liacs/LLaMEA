import numpy as np

class HybridDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim  # generally a good population size for DE
        self.crossover_rate = 0.9
        self.mutation_factor = 0.8
        self.local_search_prob = 0.3
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Initialize population
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += self.population_size
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                # Mutation
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                # Fitness-proportional mutation factor
                mutation_factor_adaptive = (self.mutation_factor * (fitness[i] / max(fitness)))
                mutant = np.clip(a + mutation_factor_adaptive * (b - c), lb, ub)

                # Dynamic crossover rate
                current_crossover_rate = self.crossover_rate * (1 - (self.evaluations / self.budget))
                cross_points = np.random.rand(self.dim) < current_crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                self.evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Update best solution
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                # Adaptive local search probability
                local_search_prob_adaptive = self.local_search_prob * (1 - (trial_fitness / best_fitness))
                if np.random.rand() < local_search_prob_adaptive and self.evaluations < self.budget:
                    improved, trial_fitness = self.local_search(trial, func, lb, ub)
                    if improved:
                        population[i] = trial
                        fitness[i] = trial_fitness
                        if trial_fitness < best_fitness:
                            best_solution = trial
                            best_fitness = trial_fitness

                # Introduce diversity mechanism
                if np.random.rand() < 0.05:  # 5% chance to introduce diversity
                    population[i] = np.random.uniform(lb, ub, self.dim)

        return best_solution

    def local_search(self, solution, func, lb, ub):
        improved = False
        step_size = (ub - lb) * 0.01  # A small step size relative to bounds
        for _ in range(5):  # Perform up to 5 local search steps
            candidate = solution + np.random.uniform(-step_size, step_size, self.dim) * 0.5  # Adaptive step size
            candidate = np.clip(candidate, lb, ub)
            candidate_fitness = func(candidate)
            self.evaluations += 1
            if candidate_fitness < func(solution):
                solution = candidate
                improved = True
        return improved, candidate_fitness