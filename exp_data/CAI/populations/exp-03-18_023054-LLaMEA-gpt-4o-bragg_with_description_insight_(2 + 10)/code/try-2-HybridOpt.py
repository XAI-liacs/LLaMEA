import numpy as np
from scipy.optimize import minimize

class HybridOpt:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = None
        self.ub = None

    def initialize_population(self, population_size):
        return np.random.uniform(self.lb, self.ub, (population_size, self.dim))

    def differential_evolution(self, pop, func, F=0.5, CR=0.9, adapt_CR=False):
        if adapt_CR:  # Adaptive crossover rate
            CR = 0.9 - 0.8 * (evaluations / self.budget)
        new_pop = np.copy(pop)
        for i in range(pop.shape[0]):
            candidates = list(range(pop.shape[0]))
            candidates.remove(i)
            a, b, c = pop[np.random.choice(candidates, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), self.lb, self.ub)
            cross_points = np.random.rand(self.dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, pop[i])
            if func(trial) > func(pop[i]):  # Maximize
                new_pop[i] = trial
        return new_pop

    def impose_periodicity(self, pop, period_length):
        for i in range(pop.shape[0]):
            pop[i] = np.tile(pop[i][:period_length], self.dim // period_length + 1)[:self.dim]
        return pop

    def local_optimization(self, x, func):
        result = minimize(lambda x: -func(x), x, bounds=[(self.lb[i], self.ub[i]) for i in range(self.dim)], method='L-BFGS-B')
        return result.x

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        population_size = 20
        period_length = self.dim // 2
        pop = self.initialize_population(population_size)
        best_solution = max(pop, key=func)  # Elitism: track best solution
        evaluations = 0

        while evaluations < self.budget:
            pop = self.differential_evolution(pop, func, adapt_CR=True)
            pop = self.impose_periodicity(pop, period_length)
            for i in range(pop.shape[0]):
                pop[i] = self.local_optimization(pop[i], func)
                if func(pop[i]) > func(best_solution):  # Update best solution
                    best_solution = pop[i]
                evaluations += 1
                if evaluations >= self.budget:
                    break

        return best_solution