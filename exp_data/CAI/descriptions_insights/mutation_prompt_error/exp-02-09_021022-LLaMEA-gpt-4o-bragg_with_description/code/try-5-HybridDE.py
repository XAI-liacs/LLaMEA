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
                self.mutation_factor = np.random.uniform(0.5, 1.0)  # Dynamic adjustment of mutation factor
                mutant = np.clip(a + self.mutation_factor * (b - c), lb, ub)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.crossover_rate
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

                # Local search
                if np.random.rand() < self.local_search_prob and self.evaluations < self.budget:
                    improved, trial_fitness = self.local_search(trial, func, lb, ub)
                    if improved:
                        population[i] = trial
                        fitness[i] = trial_fitness
                        if trial_fitness < best_fitness:
                            best_solution = trial
                            best_fitness = trial_fitness

        return best_solution

    def local_search(self, solution, func, lb, ub):
        improved = False
        step_size = (ub - lb) * 0.01  # A small step size relative to bounds
        for _ in range(5):  # Perform up to 5 local search steps
            candidate = solution + np.random.uniform(-step_size, step_size, self.dim)
            candidate = np.clip(candidate, lb, ub)
            candidate_fitness = func(candidate)
            self.evaluations += 1
            if candidate_fitness < func(solution):
                solution = candidate
                improved = True
        return improved, candidate_fitness