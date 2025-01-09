import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=100):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.F = 0.5  # Initial mutation factor
        self.CR = 0.9  # Initial crossover probability

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.pop_size

        while eval_count < self.budget:
            trial_population = np.copy(population)
            for i in range(self.pop_size):
                indices = [j for j in range(self.pop_size) if j != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = population[a] + self.F * (population[b] - population[c])
                mutant = np.clip(mutant, lb, ub)
                
                crossover = np.random.rand(self.dim) < self.CR
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial_population[i] = np.where(crossover, mutant, population[i])
                
                trial_fitness = func(trial_population[i])
                eval_count += 1
                
                if trial_fitness < fitness[i]:
                    population[i] = trial_population[i]
                    fitness[i] = trial_fitness
                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial_population[i]
            
            if eval_count % (self.pop_size * 5) == 0:  # Adjust parameters adaptively
                self.F = np.random.uniform(0.4, 0.9)
                self.CR = np.random.uniform(0.75, 1.0)

        return self.f_opt, self.x_opt