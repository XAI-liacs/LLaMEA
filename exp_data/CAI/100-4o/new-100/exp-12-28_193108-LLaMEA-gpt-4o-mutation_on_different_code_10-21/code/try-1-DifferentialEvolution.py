import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=30, F=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.f_opt = np.Inf
        self.x_opt = None
        self.success_rate = 0.2  # Initial success rate

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.pop_size

        # Optimization loop
        while eval_count < self.budget:
            new_population = np.empty_like(population)
            for i in range(self.pop_size):
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), func.bounds.lb, func.bounds.ub)
                
                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR * (1 + self.success_rate)
                if not np.any(crossover_mask):
                    crossover_mask[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover_mask, mutant, population[i])
                
                # Selection
                f_trial = func(trial)
                eval_count += 1
                
                if f_trial < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = f_trial
                    self.success_rate = min(1, self.success_rate + 0.1)  # Increment success rate
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                else:
                    new_population[i] = population[i]
                    self.success_rate = max(0, self.success_rate - 0.05)  # Decrease success rate
                    
                if eval_count >= self.budget:
                    break

            population = new_population

        return self.f_opt, self.x_opt