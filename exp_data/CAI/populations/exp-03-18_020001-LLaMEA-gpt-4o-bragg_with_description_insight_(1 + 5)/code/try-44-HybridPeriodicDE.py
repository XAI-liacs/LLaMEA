import numpy as np

class HybridPeriodicDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.cr = 0.9  # Crossover probability
        self.f = 0.8   # Differential weight
        self.periodicity_weight = 0.1  # Weight for periodicity enforcement
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        eval_count = self.population_size
        
        while eval_count < self.budget:
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = self.mutate(a, b, c, lb, ub)
                trial = self.crossover(population[i], mutant, best)
                trial = self.enforce_periodicity(trial, best)

                trial_fitness = func(trial)
                eval_count += 1
                
                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    population[i] = trial
                    if trial_fitness < fitness[best_idx]:
                        best_idx = i
                        best = trial
                else:
                    trial = self.local_search(trial, trial_fitness, func, lb, ub)
                    trial_fitness = func(trial)
                    eval_count += 1
                    if trial_fitness < fitness[i]:
                        fitness[i] = trial_fitness
                        population[i] = trial
                        if trial_fitness < fitness[best_idx]:
                            best_idx = i
                            best = trial

        return best

    def mutate(self, a, b, c, lb, ub):
        f_local = self.f * (0.5 + 0.5 * np.random.rand())  # Localized adaptive mutation
        mutant = np.clip(a + f_local * (b - c), lb, ub)
        return mutant

    def crossover(self, target, mutant, best):
        cr_local = self.cr * (0.5 + 0.5 * np.random.rand())  # Localized adaptive crossover
        cross_points = np.random.rand(self.dim) < cr_local
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        return np.where(cross_points, mutant, target)

    def enforce_periodicity(self, solution, best):
        period = 2
        for start in range(0, self.dim, period):
            end = min(start + period, self.dim)
            avg = (solution[start:end] + best[start:end] + np.roll(solution, period)[start:end]) / 3
            solution[start:end] = np.mean(avg)
        return solution

    def local_search(self, solution, fitness, func, lb, ub):
        step_size = 0.05 * (ub - lb)
        candidate = np.clip(solution + step_size * np.random.uniform(-1, 1, self.dim), lb, ub)
        if func(candidate) < fitness:
            return candidate
        step_size *= 0.5
        candidate = np.clip(solution + step_size * np.random.uniform(-1, 1, self.dim), lb, ub)
        return candidate if func(candidate) < fitness else solution