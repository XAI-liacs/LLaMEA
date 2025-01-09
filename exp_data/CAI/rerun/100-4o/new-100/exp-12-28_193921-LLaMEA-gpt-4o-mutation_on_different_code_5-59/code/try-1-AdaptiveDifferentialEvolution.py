import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.pop_size = 10 * dim
        self.scale_factor = 0.5
        self.crossover_rate = 0.9
        self.mutation_factor = 0.8

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        population = np.random.uniform(bounds[0], bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.pop_size

        while eval_count < self.budget:
            for i in range(self.pop_size):
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.scale_factor * (b - c), bounds[0], bounds[1])
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(cross_points, mutant, population[i])
                
                f_trial = func(trial)
                eval_count += 1
                
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    
                    # Dynamic adaptation of control parameters
                    self.scale_factor = np.clip(self.scale_factor + np.random.normal(0, 0.1), 0.4, 1.0)
                    self.crossover_rate = np.clip(self.crossover_rate + np.random.uniform(-0.05, 0.05), 0.6, 1.0)

            best_idx = np.argmin(fitness)
            if fitness[best_idx] < self.f_opt:
                self.f_opt = fitness[best_idx]
                self.x_opt = population[best_idx]

            # Adjust mutation factor based on population diversity
            diversity = np.std(population, axis=0).mean()
            self.mutation_factor = np.clip(0.5 + diversity / 10.0, 0.4, 1.0)

        return self.f_opt, self.x_opt