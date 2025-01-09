import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20 * dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.bounds = (-5.0, 5.0)
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, dim))
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9

    def mutate(self, target_idx):
        indices = [idx for idx in range(self.pop_size) if idx != target_idx]
        a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
        f = np.random.uniform(0.4, 0.9)  # Adaptive mutation factor
        mutant = np.clip(a + f * (b - c), self.bounds[0], self.bounds[1])
        return mutant

    def crossover(self, target, mutant):
        cr = np.random.uniform(0.6, 1.0) if np.random.rand() < 0.5 else 0.9  # Dynamic crossover rate
        crossover_mask = np.random.rand(self.dim) < cr
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def __call__(self, func):
        evaluations = 0
        for _ in range(self.budget // self.pop_size):
            for i in range(self.pop_size):
                target = self.population[i]
                mutant = self.mutate(i)
                trial = self.crossover(target, mutant)

                f_target = func(target)
                f_trial = func(trial)
                evaluations += 2
                
                if f_trial < f_target:
                    self.population[i] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

                if evaluations >= self.budget:
                    return self.f_opt, self.x_opt

        return self.f_opt, self.x_opt