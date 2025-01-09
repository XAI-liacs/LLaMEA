import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = min(20, self.budget // 10)
        self.mutation_factor = 0.8
        self.crossover_prob = 0.9
        self.bounds = [-5.0, 5.0]

    def __call__(self, func):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.population_size
        
        while eval_count < self.budget:
            fitness_mean = np.mean(fitness)
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                self.mutation_factor = 0.5 + 0.5 * (fitness[i] - fitness_mean) / np.ptp(fitness)
                mutant = np.clip(a + self.mutation_factor * (b - c), self.bounds[0], self.bounds[1])
                self.crossover_prob = 0.5 + 0.5 * (fitness[i] - fitness_mean) / np.ptp(fitness)
                cross_points = np.random.rand(self.dim) < self.crossover_prob
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                f_trial = func(trial)
                eval_count += 1
                
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial
                
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

                if eval_count >= self.budget:
                    break

        return self.f_opt, self.x_opt