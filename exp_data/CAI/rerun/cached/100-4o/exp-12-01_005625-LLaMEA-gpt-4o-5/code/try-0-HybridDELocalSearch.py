import numpy as np

class HybridDELocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim  # Population size
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.lb = -5.0  # Lower bound
        self.ub = 5.0  # Upper bound
        self.local_search_iters = 5  # Local Search Iterations

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = self.pop_size

        while evals < self.budget:
            # Differential Evolution Mutation
            for i in range(self.pop_size):
                indices = np.random.choice(self.pop_size, 3, replace=False)
                x0, x1, x2 = population[indices]
                mutant = np.clip(x0 + self.F * (x1 - x2), self.lb, self.ub)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, population[i])
                
                # Local Search - Simple Hill Climbing
                trial = self.local_search(func, trial, evals)
                evals += self.local_search_iters

                # Selection
                trial_fitness = func(trial)
                evals += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                if evals >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]

    def local_search(self, func, candidate, current_evals):
        step_size = 0.1
        for _ in range(self.local_search_iters):
            if current_evals >= self.budget:
                break
            candidate_eval = func(candidate)
            for j in range(self.dim):
                if current_evals >= self.budget:
                    break
                perturbed_candidate = np.copy(candidate)
                perturbed_candidate[j] += step_size * (np.random.rand() - 0.5) * 2
                perturbed_candidate = np.clip(perturbed_candidate, self.lb, self.ub)
                new_eval = func(perturbed_candidate)
                current_evals += 1
                if new_eval < candidate_eval:
                    candidate = perturbed_candidate
                    candidate_eval = new_eval
        return candidate