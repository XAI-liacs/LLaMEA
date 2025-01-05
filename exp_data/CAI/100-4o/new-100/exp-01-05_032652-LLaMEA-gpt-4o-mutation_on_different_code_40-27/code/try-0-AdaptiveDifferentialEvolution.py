import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=20):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lower_bound, upper_bound = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lower_bound, upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size

        while eval_count < self.budget:
            diversity = np.std(population, axis=0).mean()
            F = 0.5 + 0.5 * (1 - diversity / (upper_bound - lower_bound).mean())
            CR = 0.5 + 0.5 * (1 - diversity / (upper_bound - lower_bound).mean())
            
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break
                
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                
                mutant = population[a] + F * (population[b] - population[c])
                mutant = np.clip(mutant, lower_bound, upper_bound)
                
                crossover = np.random.rand(self.dim) < CR
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                
                trial = np.where(crossover, mutant, population[i])
                
                trial_fitness = func(trial)
                eval_count += 1
                
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    
                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial

        return self.f_opt, self.x_opt