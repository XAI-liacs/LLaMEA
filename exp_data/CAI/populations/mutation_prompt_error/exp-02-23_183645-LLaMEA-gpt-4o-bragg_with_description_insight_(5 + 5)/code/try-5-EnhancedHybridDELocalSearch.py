import numpy as np
from scipy.optimize import minimize

class EnhancedHybridDELocalSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.crossover_rate = 0.7
        self.differential_weight = 0.8
        self.base_local_search_probability = 0.2

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        fitness = np.array([func(ind) for ind in pop])
        evaluations = self.population_size

        best_idx = np.argmin(fitness)
        best = pop[best_idx]
        best_fitness = fitness[best_idx]

        while evaluations < self.budget:
            new_pop = np.copy(pop)
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.differential_weight * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    new_pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best = trial
                        best_fitness = trial_fitness

            pop = new_pop

            # Dynamically adjust local search probability and apply local search
            local_search_probability = self.base_local_search_probability * (1 - evaluations / self.budget)
            if np.random.rand() < local_search_probability:
                for i in range(self.population_size // 5):
                    if evaluations >= self.budget:
                        break
                    local_search_result = minimize(func, pop[i], bounds=list(zip(lb, ub)), method='L-BFGS-B')
                    evaluations += local_search_result.nfev
                    if local_search_result.fun < fitness[i]:
                        pop[i] = local_search_result.x
                        fitness[i] = local_search_result.fun
                        if local_search_result.fun < best_fitness:
                            best = local_search_result.x
                            best_fitness = local_search_result.fun                      

        return best