import numpy as np

class HDE_APC_LS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.population_size = min(100, 10 * dim)
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.best_solution = None
        self.best_fitness = np.inf
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])
                if self.fitness[i] < self.best_fitness:
                    self.best_fitness = self.fitness[i]
                    self.best_solution = self.population[i].copy()

    def select_parents(self, i):
        indices = list(range(self.population_size))
        indices.remove(i)
        return np.random.choice(indices, 3, replace=False)

    def mutate(self, i):
        a, b, c = self.select_parents(i)
        mutant = self.population[a] + self.F * (self.population[b] - self.population[c])
        mutant = np.clip(mutant, self.bounds[0], self.bounds[1])
        return mutant

    def crossover(self, target, mutant):
        trial = np.copy(target)
        for j in range(self.dim):
            if np.random.rand() < self.CR or j == np.random.randint(self.dim):
                trial[j] = mutant[j]
        return trial

    def local_search(self, target):
        perturbation = np.random.normal(0, 0.1, self.dim)
        candidate = np.clip(target + perturbation, self.bounds[0], self.bounds[1])
        return candidate

    def __call__(self, func):
        self.evaluate_population(func)
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                mutant = self.mutate(i)
                trial = self.crossover(self.population[i], mutant)
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.best_solution = trial.copy()

                if evaluations < self.budget:
                    local_candidate = self.local_search(self.population[i])
                    local_fitness = func(local_candidate)
                    evaluations += 1
                    if local_fitness < self.fitness[i]:
                        self.population[i] = local_candidate
                        self.fitness[i] = local_fitness
                        if local_fitness < self.best_fitness:
                            self.best_fitness = local_fitness
                            self.best_solution = local_candidate.copy()

            # Adaptive parameter control
            self.F = 0.4 + 0.3 * np.random.rand()
            self.CR = 0.8 + 0.2 * np.random.rand()

        return self.best_solution