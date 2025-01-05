import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.cr = 0.5  # Crossover rate
        self.f = 0.8  # Differential weight
        self.strategy_adaptation = True
        self.strategy_adaptation_rate = 0.05  # Adaptation rate for strategies
        self.strategy_weights = np.random.uniform(0, 2, size=budget)  # Initial strategy weights

    def __call__(self, func):
        pop_size = 10 * self.dim
        population = np.random.uniform(-5.0, 5.0, size=(pop_size, self.dim))

        for i in range(self.budget):
            for j in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]

                if self.strategy_adaptation:
                    f = np.clip(self.f + np.random.normal(0, self.strategy_adaptation_rate), 0, 2)
                    cr = np.clip(self.cr + np.random.normal(0, self.strategy_adaptation_rate), 0, 1)
                else:
                    f = self.f
                    cr = self.cr

                mutant = np.clip(a + f * (b - c), -5.0, 5.0)
                crossover = np.random.rand(self.dim) < cr
                trial = np.where(crossover, mutant, population[j])

                f_trial = func(trial)
                if f_trial < func(population[j]):
                    population[j] = trial

                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

            self.strategy_weights[i] = f

        return self.f_opt, self.x_opt