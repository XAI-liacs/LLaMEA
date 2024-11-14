import numpy as np

class EnhancedHybridPSODE:
    def __init__(self, budget, dim, pop_size=30, w=0.5, c1=1.5, c2=1.5, f=0.5, cr=0.9, adapt_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.f = f
        self.cr = cr
        self.adapt_rate = adapt_rate
        self.mut_prob = 0.5

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))

        population = initialize_population()
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]

        for _ in range(self.budget - self.pop_size):
            for i in range(self.pop_size):
                r1, r2, r3 = np.random.choice(self.pop_size, 3, replace=False)
                mutant = population[r1] + self.f * (population[r2] - population[r3])
                self.f = max(0.1, min(0.9, self.f + np.random.normal(0, self.adapt_rate)))
                crossover = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover, mutant, population[i])
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < fitness[best_idx]:
                        best_idx = i
                        best_solution = trial

            for i in range(self.pop_size):
                r1, r2 = np.random.choice(self.pop_size, 2, replace=False)
                v = self.w * population[i] + self.c1 * np.random.rand(self.dim) * (best_solution - population[i]) + self.c2 * np.random.rand(self.dim) * (population[r1] - population[r2])
                mutation_direction = np.random.choice([-1, 1], p=[self.mut_prob, 1 - self.mut_prob])
                population[i] = np.clip(v, -5.0, 5.0)
                
            # Dynamic population resizing
            if np.random.rand() < 0.1:  # 2% code change
                self.pop_size = int(min(50, self.pop_size * 1.1))  # Increase population size up to a limit

        return best_solution