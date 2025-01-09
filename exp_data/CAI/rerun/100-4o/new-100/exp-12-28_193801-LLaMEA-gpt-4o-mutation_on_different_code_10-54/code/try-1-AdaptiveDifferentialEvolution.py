import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        population = np.random.uniform(bounds[0], bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.f_opt, self.x_opt = np.min(fitness), population[np.argmin(fitness)]
        eval_count = self.pop_size

        while eval_count < self.budget:
            for i in range(self.pop_size):
                indices = np.random.choice(list(range(i)) + list(range(i+1, self.pop_size)), 3, replace=False)
                a, b, c = population[indices]
                
                # Adaptive mutation factor
                F_dynamic = self.F + 0.1 * np.random.rand()
                
                mutant_vector = np.clip(a + F_dynamic * (b - c), bounds[0], bounds[1])
                
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial_vector = np.where(crossover_mask, mutant_vector, population[i])
                
                trial_fitness = func(trial_vector)
                eval_count += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = trial_fitness
                    if trial_fitness < self.f_opt:
                        self.f_opt, self.x_opt = trial_fitness, trial_vector
                
                if eval_count >= self.budget:
                    break

        return self.f_opt, self.x_opt