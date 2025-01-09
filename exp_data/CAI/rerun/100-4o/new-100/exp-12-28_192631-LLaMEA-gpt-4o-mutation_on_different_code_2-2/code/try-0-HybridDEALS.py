import numpy as np

class HybridDEALS:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.func_evals = 0

    def differential_evolution(self, population, func):
        np.random.shuffle(population)
        for i in range(self.population_size):
            if self.func_evals >= self.budget:
                break
                
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + self.mutation_factor * (b - c), func.bounds.lb, func.bounds.ub)
            cross_points = np.random.rand(self.dim) < self.crossover_rate
            trial = np.where(cross_points, mutant, population[i])
            f_trial = func(trial)
            self.func_evals += 1
            
            if f_trial < self.fitness[i]:
                self.fitness[i] = f_trial
                population[i] = trial
                if f_trial < self.f_opt:
                    self.f_opt, self.x_opt = f_trial, trial

    def local_search(self, x, func, step_size=0.1, max_iter=10):
        best_x = x
        best_f = func(x)
        self.func_evals += 1
        for _ in range(max_iter):
            if self.func_evals >= self.budget:
                break
            candidate = best_x + np.random.uniform(-step_size, step_size, size=self.dim)
            candidate = np.clip(candidate, func.bounds.lb, func.bounds.ub)
            f_candidate = func(candidate)
            self.func_evals += 1
            if f_candidate < best_f:
                best_f, best_x = f_candidate, candidate
                if best_f < self.f_opt:
                    self.f_opt, self.x_opt = best_f, best_x
        return best_x, best_f

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        self.fitness = np.array([func(ind) for ind in population])
        self.func_evals += self.population_size
        
        if np.min(self.fitness) < self.f_opt:
            self.f_opt = np.min(self.fitness)
            self.x_opt = population[np.argmin(self.fitness)]

        while self.func_evals < self.budget:
            self.differential_evolution(population, func)
            for i in range(self.population_size):
                if self.func_evals >= self.budget:
                    break
                if np.random.rand() < 0.1:  # probability to perform local search
                    population[i], self.fitness[i] = self.local_search(population[i], func)
        
        return self.f_opt, self.x_opt