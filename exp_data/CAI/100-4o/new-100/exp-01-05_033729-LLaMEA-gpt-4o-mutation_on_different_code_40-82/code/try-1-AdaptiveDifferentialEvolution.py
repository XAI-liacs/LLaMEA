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

            # Dynamic population resizing
            if eval_count % (self.pop_size * 2) == 0:
                new_size = max(4, int(self.pop_size * (0.9 + 0.1 * np.random.rand())))
                population = population[:new_size]
                fitness = fitness[:new_size]

            # Feedback-based parameter tuning
            F_success = np.mean(fitness < (self.f_opt + 1e-8))
            self.F = 0.5 + 0.4 * F_success
            self.CR = 0.7 + 0.2 * (1 - F_success)

        return self.f_opt, self.x_opt