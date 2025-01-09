import numpy as np

class HybridAnnealingDE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 20
        self.cr = 0.9  # Crossover probability
        self.f = 0.8   # Differential weight
        self.init_temp = 10.0
        self.min_temp = 0.1

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        min_idx = np.argmin(fitness)
        self.f_opt = fitness[min_idx]
        self.x_opt = population[min_idx]
        temp = self.init_temp

        evaluations = self.population_size
        best_solution = population[min_idx]
        while evaluations < self.budget:
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                adaptive_f = self.f * np.random.uniform(0.5, 1.5)
                mutant = np.clip(a + adaptive_f * (b - c), func.bounds.lb, func.bounds.ub)
                crossover = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover, mutant, population[i])
                
                f_trial = func(trial)
                delta_f = f_trial - fitness[i]
                if delta_f < 0 or np.exp(-delta_f / temp) > np.random.rand():
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        best_solution = trial
                evaluations += 1
                if evaluations >= self.budget:
                    break
            temp = max(temp * 0.9, self.min_temp)
            population[np.argmax(fitness)] = best_solution  # Elitism step

        return self.f_opt, self.x_opt