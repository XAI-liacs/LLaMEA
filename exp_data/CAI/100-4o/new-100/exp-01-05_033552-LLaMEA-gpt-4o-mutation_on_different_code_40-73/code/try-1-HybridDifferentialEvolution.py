import numpy as np

class HybridDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.pop_size = 10 * dim
        self.mutation_factor = 0.5 + np.random.rand() * 0.3  # Self-adaptive mutation factor
        self.crossover_rate = 0.8 + np.random.rand() * 0.2  # Self-adaptive crossover rate
        self.local_search_prob = 0.1
        self.bounds = [-5.0, 5.0]

    def __call__(self, func):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_used = self.pop_size
        
        while budget_used < self.budget:
            for i in range(self.pop_size):
                indices = np.random.choice(np.delete(np.arange(self.pop_size), i), 3, replace=False)
                x0, x1, x2 = population[indices]
                
                mutant = x0 + self.mutation_factor * (x1 - x2)
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])
                
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                
                trial = np.where(cross_points, mutant, population[i])
                
                if np.random.rand() < self.local_search_prob:
                    trial = self.adaptive_local_search(trial, func)
                
                f_trial = func(trial)
                budget_used += 1
                
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial
                
                if budget_used >= self.budget:
                    break

        return self.f_opt, self.x_opt

    def adaptive_local_search(self, x, func):
        step_size = 0.1
        for _ in range(5):
            x_new = x + np.random.uniform(-step_size, step_size, self.dim)
            x_new = np.clip(x_new, self.bounds[0], self.bounds[1])
            if func(x_new) < func(x):
                x = x_new
                step_size *= 1.2  # Increase step size if improvement is found
            else:
                step_size *= 0.8  # Decrease step size if no improvement
        return x