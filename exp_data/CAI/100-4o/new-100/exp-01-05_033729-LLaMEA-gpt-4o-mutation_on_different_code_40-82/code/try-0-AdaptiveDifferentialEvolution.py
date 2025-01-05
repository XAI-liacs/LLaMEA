import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.F = 0.8  # Default mutation factor
        self.CR = 0.9  # Default crossover probability

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.pop_size

        while eval_count < self.budget:
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant_vector = np.clip(a + self.F * (b - c), lb, ub)
                crossover = np.random.rand(self.dim) < self.CR
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial_vector = np.where(crossover, mutant_vector, population[i])
                trial_fitness = func(trial_vector)
                eval_count += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = trial_fitness
                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial_vector

                if eval_count >= self.budget:
                    break

            # Adaptive mechanism to adjust F and CR based on population diversity
            diversity = np.mean(np.std(population, axis=0)) / (ub - lb)
            self.F = max(0.5, min(0.9, 1.2 - diversity))
            self.CR = max(0.5, min(0.9, 0.7 + diversity))

        return self.f_opt, self.x_opt