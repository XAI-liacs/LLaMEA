import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = None

    def _initialize_population(self, size, lb, ub):
        return np.random.uniform(lb, ub, (size, self.dim))

    def _apply_periodicity(self, solutions):
        period = self.dim // 2
        for sol in solutions:
            for i in range(period):
                sol[i + period] = sol[i]  # Enforce periodicity
        return solutions

    def _local_search(self, solution, func):
        result = minimize(func, solution, method='L-BFGS-B', bounds=self.bounds)
        return result.x

    def __call__(self, func):
        self.bounds = list(zip(func.bounds.lb, func.bounds.ub))
        pop_size = 10  # Size of the population
        evaluations = 0

        # Step 1: Initialize population
        population = self._initialize_population(pop_size, func.bounds.lb, func.bounds.ub)

        # Step 2: Encourage periodicity in the initial population
        population = self._apply_periodicity(population)

        best_solution = None
        best_score = np.inf

        while evaluations < self.budget:
            # Evaluate the population
            scores = np.array([func(ind) for ind in population])
            evaluations += pop_size

            # Update best solution
            min_idx = np.argmin(scores)
            if scores[min_idx] < best_score:
                best_score = scores[min_idx]
                best_solution = population[min_idx]

            # Step 3: Differential Evolution-like operation
            new_population = []
            for i in range(pop_size):
                indices = np.random.choice(pop_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + 0.8 * (b - c), func.bounds.lb, func.bounds.ub)  # Corrected line
                trial = np.where(np.random.rand(self.dim) < 0.9, mutant, population[i])  # Crossover rate CR=0.9
                new_population.append(trial)

            # Step 4: Encourage periodicity in the new population
            new_population = self._apply_periodicity(new_population)

            # Step 5: Local optimization on best candidate
            if evaluations + 1 <= self.budget:
                best_solution = self._local_search(best_solution, func)
                best_score = func(best_solution)
                evaluations += 1

            # Prepare for next iteration
            population = new_population

        return best_solution