import numpy as np

class HybridDifferentialEvolutionSA:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.f_opt = np.inf
        self.x_opt = None
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.temperature = 1.0
        self.cooling_rate = 0.99

    def differential_mutation(self, idx):
        indices = np.random.choice([i for i in range(self.population_size) if i != idx], 3, replace=False)
        a, b, c = self.population[indices]
        mutant = np.clip(a + self.mutation_factor * (b - c), -5.0, 5.0)
        return mutant

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def __call__(self, func):
        evals = 0
        for i in range(self.population_size):
            self.fitness[i] = func(self.population[i])
            evals += 1
            if self.fitness[i] < self.f_opt:
                self.f_opt = self.fitness[i]
                self.x_opt = self.population[i]

        while evals < self.budget:
            for i in range(self.population_size):
                mutant = self.differential_mutation(i)
                trial = self.crossover(self.population[i], mutant)
                f_trial = func(trial)
                evals += 1

                # Simulated Annealing acceptance criterion
                if f_trial < self.fitness[i] or np.exp((self.fitness[i] - f_trial) / self.temperature) > np.random.rand():
                    self.population[i] = trial
                    self.fitness[i] = f_trial

                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

                if evals >= self.budget:
                    break

            self.temperature *= self.cooling_rate

        return self.f_opt, self.x_opt