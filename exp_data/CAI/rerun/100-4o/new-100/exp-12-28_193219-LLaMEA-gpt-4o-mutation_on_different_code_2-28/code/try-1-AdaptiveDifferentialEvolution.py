import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 10 * dim
        self.F = 0.5  # Mutation factor
        self.CR = 0.9  # Crossover probability

    def __call__(self, func):
        np.random.seed(42)
        bounds = np.array([func.bounds.lb, func.bounds.ub])
        population = np.random.uniform(bounds[0], bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        for evals in range(self.budget - self.population_size):
            idxs = np.arange(self.population_size)
            np.random.shuffle(idxs)
            
            for i in range(self.population_size):
                a, b, c = population[idxs[i]], population[idxs[(i + 1) % len(idxs)]], population[idxs[(i + 2) % len(idxs)]]
                scale_factor = np.random.uniform(0.5, 1.5)  # Nonlinear scaling factor
                mutant = np.clip(a + scale_factor * self.F * (b - c), bounds[0], bounds[1])
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, population[i])
                
                f_trial = func(trial)
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                        
            successful_mutations = fitness < np.percentile(fitness, 25)
            if np.any(successful_mutations):
                scale = np.mean(np.linalg.norm(population[successful_mutations], axis=1))
                self.F = 0.5 * (np.tanh(scale / 20.0) + 1.0)

        return self.f_opt, self.x_opt