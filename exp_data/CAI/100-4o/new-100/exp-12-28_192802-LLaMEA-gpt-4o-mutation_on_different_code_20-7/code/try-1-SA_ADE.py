import numpy as np

class SA_ADE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.bounds = (-5.0, 5.0)
        self.temp_init = 100
        self.temp_final = 0.1
        self.temperature = self.temp_init
        self.F = 0.5  # Starting mutation factor

    def _acceptance_probability(self, delta, temperature):
        return np.exp(-delta / temperature)

    def _mutate(self, x):
        idxs = np.random.choice(np.arange(self.dim), size=3, replace=False)
        a, b, c = x[idxs]
        mutant = a + self.F * (b - c)
        return np.clip(mutant, *self.bounds)

    def __call__(self, func):
        x = np.random.uniform(*self.bounds, size=self.dim)
        self.f_opt = func(x)
        self.x_opt = x.copy()

        for i in range(self.budget):
            if self.temperature > self.temp_final:
                candidate = x + self._mutate(x)
                candidate_fitness = func(candidate)

                delta_f = candidate_fitness - self.f_opt
                if candidate_fitness < self.f_opt or np.random.rand() < self._acceptance_probability(delta_f, self.temperature):
                    x = candidate
                    if candidate_fitness < self.f_opt:
                        self.f_opt = candidate_fitness
                        self.x_opt = candidate
                        self.F = min(1.0, self.F + 0.1)  # Increase F upon improvement
                    else:
                        self.F = max(0.1, self.F - 0.05)  # Decrease F otherwise

                self.temperature *= (self.temp_final / self.temp_init) ** (1.0 / self.budget)
            else:
                break

        return self.f_opt, self.x_opt