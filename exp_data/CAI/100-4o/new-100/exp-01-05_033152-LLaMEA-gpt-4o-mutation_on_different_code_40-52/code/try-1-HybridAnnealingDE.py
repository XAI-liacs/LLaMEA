import numpy as np

class HybridAnnealingDE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 20
        self.T_init = 1.0
        self.T_final = 0.001
        self.mutation_adaptation_rate = 0.1

    def adaptive_mutation_factor(self, f_old, f_new, F):
        if f_new < f_old:
            F = min(1.0, F + self.mutation_adaptation_rate)
        else:
            F = max(0.4, F - self.mutation_adaptation_rate)
        return F

    def differential_evolution_step(self, population, func):
        F = 0.8
        CR = 0.9
        new_population = np.empty_like(population)
        
        for i in range(self.population_size):
            indices = list(range(self.population_size))
            indices.remove(i)
            a, b, c = np.random.choice(indices, 3, replace=False)
            mutant = population[a] + F * (population[b] - population[c])
            mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
            
            cross_points = np.random.rand(self.dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            
            trial = np.where(cross_points, mutant, population[i])
            
            f_trial = func(trial)
            f_target = func(population[i])
            
            if f_trial < f_target:
                new_population[i] = trial
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
            else:
                new_population[i] = population[i]

            F = self.adaptive_mutation_factor(f_target, f_trial, F)

        return new_population

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        func_values = np.apply_along_axis(func, 1, population)
        
        self.f_opt = np.min(func_values)
        self.x_opt = population[np.argmin(func_values)]
        
        T = self.T_init
        evals = self.population_size

        while evals < self.budget:
            population = self.differential_evolution_step(population, func)
            evals += self.population_size

            if np.random.rand() < np.exp((self.f_opt - func(self.x_opt)) / T):
                func_values = np.apply_along_axis(func, 1, population)
                best_index = np.argmin(func_values)
                if func_values[best_index] < self.f_opt:
                    self.f_opt = func_values[best_index]
                    self.x_opt = population[best_index]
            
            T = max(self.T_final, T * 0.9)

        return self.f_opt, self.x_opt