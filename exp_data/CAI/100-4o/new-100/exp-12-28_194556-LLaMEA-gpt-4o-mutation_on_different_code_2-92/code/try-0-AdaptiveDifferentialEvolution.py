import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=50, f=0.5, cr=0.7):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.f = f
        self.cr = cr
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        bounds = [func.bounds.lb, func.bounds.ub]
        population = np.random.uniform(bounds[0], bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        
        self.f_opt, best_idx = np.min(fitness), np.argmin(fitness)
        self.x_opt = population[best_idx]

        eval_count = self.population_size

        while eval_count < self.budget:
            new_population = np.copy(population)
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + self.f * (b - c), bounds[0], bounds[1])
                crossover_mask = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover_mask, mutant, population[i])

                f_trial = func(trial)
                eval_count += 1
                
                if f_trial < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                
                if eval_count >= self.budget:
                    break
            
            population = new_population

            # Adaptive parameters based on diversity
            diversity = np.std(population, axis=0).mean()
            self.f = 0.5 + 0.3 * (1 - diversity / (bounds[1] - bounds[0]).mean())
            self.cr = 0.9 - 0.4 * (1 - diversity / (bounds[1] - bounds[0]).mean())
        
        return self.f_opt, self.x_opt