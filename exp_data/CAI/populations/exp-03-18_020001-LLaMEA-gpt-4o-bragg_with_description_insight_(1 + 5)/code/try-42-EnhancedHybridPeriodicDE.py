import numpy as np

class EnhancedHybridPeriodicDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10 * dim, 50)
        self.cr = 0.85  # Crossover probability
        self.f = 0.7    # Differential weight
        self.periodicity_weight = 0.15  # Weight for periodicity reinforcement
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best = population[best_idx]
        eval_count = self.population_size
        
        while eval_count < self.budget:
            self.population_size = max(5, int(10 * (self.budget - eval_count) / self.budget))  # Dynamic resizing
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = self.mutate(a, b, c, lb, ub)
                trial = self.crossover(population[i], mutant)
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
        mutant = np.clip(a + (self.f + 0.1 * np.random.rand()) * (b - c), lb, ub)
        return mutant

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.cr
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
        step_size *= 0.7
        candidate = np.clip(solution + step_size * np.random.uniform(-1, 1, self.dim), lb, ub)
        return candidate if func(candidate) < fitness else solution