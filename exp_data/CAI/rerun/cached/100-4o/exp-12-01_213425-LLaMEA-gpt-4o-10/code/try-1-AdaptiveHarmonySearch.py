import numpy as np

class AdaptiveHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.harmony_memory_size = 30
        self.harmony_memory = np.random.uniform(self.lower_bound, self.upper_bound, (self.harmony_memory_size, self.dim))
        self.harmony_memory_fitness = np.array([float('inf')] * self.harmony_memory_size)
        self.best_solution = None
        self.best_fitness = float('inf')
        self.iteration = 0

    def levy_flight(self, beta=1.5):
        sigma = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                 (np.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / (np.abs(v)**(1 / beta))
        return step

    def __call__(self, func):
        evals = 0
        while evals < self.budget:
            new_harmony = np.copy(self.harmony_memory[np.random.randint(self.harmony_memory_size)])
            if np.random.rand() < 0.9:  # Harmony memory consideration rate
                new_harmony += self.levy_flight() * 0.05  # small step size for Lévy flight
            else:
                new_harmony = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            new_harmony = np.clip(new_harmony, self.lower_bound, self.upper_bound)

            new_fitness = func(new_harmony)
            evals += 1

            if new_fitness < np.max(self.harmony_memory_fitness):
                worst_index = np.argmax(self.harmony_memory_fitness)
                self.harmony_memory[worst_index] = new_harmony
                self.harmony_memory_fitness[worst_index] = new_fitness

            if new_fitness < self.best_fitness:
                self.best_fitness = new_fitness
                self.best_solution = new_harmony

            self.iteration += 1

        return self.best_solution