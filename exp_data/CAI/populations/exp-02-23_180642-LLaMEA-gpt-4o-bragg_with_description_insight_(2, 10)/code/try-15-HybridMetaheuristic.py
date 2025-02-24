import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.population = None
        self.evaluations = 0

    def quasi_oppositional_initialization(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        opposite_population = lb + ub - self.population
        combined_population = np.vstack((self.population, opposite_population))
        self.population = combined_population[np.random.choice(2 * self.population_size, self.population_size, replace=False)]

    def differential_evolution_step(self, func, lb, ub):
        new_population = np.empty_like(self.population)

        for i in range(self.population_size):
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + self.F * (b - c), lb, ub)

            cross_points = np.random.rand(self.dim) < self.CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True

            trial = np.where(cross_points, mutant, self.population[i])
            if func(trial) < func(self.population[i]):
                new_population[i] = trial
            else:
                new_population[i] = self.population[i]

            self.evaluations += 1
            if self.evaluations >= self.budget:
                return new_population

        return new_population

    def impose_periodicity(self, solution, period_length=3):  # Changed line 1
        for i in range(0, self.dim, period_length):
            solution[i:i+period_length] = np.mean(solution[i:i+period_length])
        return solution

    def local_search(self, func, solution, lb, ub):
        bounds = [(lb[i], ub[i]) for i in range(self.dim)]
        result = minimize(func, solution, method='L-BFGS-B', bounds=bounds)
        self.evaluations += result.nfev
        return result.x

    def adaptive_strategies(self):  # Changed line 2
        self.F = 0.4 + np.random.random() * 0.6  # Changed line 3
        self.CR = 0.8 + np.random.random() * 0.2  # Changed line 4

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.quasi_oppositional_initialization(lb, ub)

        best_solution = None
        best_value = float('inf')

        while self.evaluations < self.budget:
            self.adaptive_strategies()  # Changed line 5
            self.population = self.differential_evolution_step(func, lb, ub)
            for i in range(self.population_size):
                candidate = self.impose_periodicity(self.population[i])
                candidate = self.local_search(func, candidate, lb, ub)

                f_candidate = func(candidate)
                if f_candidate < best_value:
                    best_value = f_candidate
                    best_solution = candidate

                if self.evaluations >= self.budget:
                    break

        return best_solution