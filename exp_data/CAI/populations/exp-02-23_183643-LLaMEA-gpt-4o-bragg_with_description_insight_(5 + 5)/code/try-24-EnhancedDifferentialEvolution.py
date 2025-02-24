import numpy as np
from scipy.optimize import minimize

class EnhancedDifferentialEvolution:
    def __init__(self, budget, dim, pop_size=50, F=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.evaluations = 0

    def quasi_opposition_based_initialization(self, lb, ub):
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        pop_opposite = lb + ub - pop
        return np.vstack((pop, pop_opposite))

    def enforce_periodicity(self, candidate):
        period = self.dim // 2
        periodic_candidate = np.tile(candidate[:period], self.dim // period)
        return periodic_candidate

    def adaptive_tune_parameters(self):
        diversity = np.std(self.population, axis=0).mean()
        self.F = 0.5 + 0.3 * (diversity / (np.max([diversity, 1e-5])))
        self.CR = 0.5 + 0.4 * (1 - (diversity / (np.max([diversity, 1e-5]))))

    def optimize(self, func, lb, ub):
        lb = np.array(lb)
        ub = np.array(ub)
        self.population = self.quasi_opposition_based_initialization(lb, ub)
        population = self.population[:self.pop_size]
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += len(population)
        
        while self.evaluations < self.budget:
            self.adaptive_tune_parameters()
            for i in range(self.pop_size):
                if self.evaluations >= self.budget:
                    break
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                trial = self.enforce_periodicity(trial)
                
                f = func(trial)
                self.evaluations += 1
                
                if f < fitness[i]:
                    fitness[i] = f
                    population[i] = trial

            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]
            res = minimize(func, best_solution, method='BFGS', bounds=[(lb[i], ub[i]) for i in range(self.dim)])
            if res.fun < fitness[best_idx]:
                population[best_idx] = res.x
                fitness[best_idx] = res.fun

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        return self.optimize(func, lb, ub)