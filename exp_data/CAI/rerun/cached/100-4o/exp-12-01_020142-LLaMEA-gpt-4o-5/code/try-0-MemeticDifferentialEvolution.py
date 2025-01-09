import numpy as np

class MemeticDifferentialEvolution:
    def __init__(self, budget, dim, pop_size=50, F=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.bounds = (-5.0, 5.0)
        self.best_solution = None
        self.best_fitness = np.inf

    def _initialize_population(self):
        lower, upper = self.bounds
        return np.random.uniform(lower, upper, (self.pop_size, self.dim))

    def _evaluate_population(self, population, func):
        fitness = np.apply_along_axis(func, 1, population)
        return fitness

    def _select_parents(self, population, fitness):
        indices = np.random.choice(self.pop_size, 3, replace=False)
        p1, p2, p3 = population[indices]
        return p1, p2, p3

    def _crossover(self, target, mutant):
        trial = np.copy(target)
        for i in range(self.dim):
            if np.random.rand() < self.CR or i == np.random.randint(self.dim):
                trial[i] = mutant[i]
        return trial

    def _local_search(self, candidate, func):
        step_size = 0.01
        for i in range(self.dim):
            candidate[i] += step_size * np.random.randn()
            candidate[i] = np.clip(candidate[i], *self.bounds)
        return candidate, func(candidate)

    def __call__(self, func):
        population = self._initialize_population()
        fitness = self._evaluate_population(population, func)
        evaluations = self.pop_size

        while evaluations < self.budget:
            for i in range(self.pop_size):
                p1, p2, p3 = self._select_parents(population, fitness)
                mutant = np.clip(p1 + self.F * (p2 - p3), *self.bounds)
                trial = self._crossover(population[i], mutant)
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < self.best_fitness:
                        self.best_solution = trial
                        self.best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

            for i in range(self.pop_size):
                if evaluations < self.budget:
                    candidate, local_fitness = self._local_search(population[i], func)
                    evaluations += 1
                    if local_fitness < fitness[i]:
                        population[i] = candidate
                        fitness[i] = local_fitness

                        if local_fitness < self.best_fitness:
                            self.best_solution = candidate
                            self.best_fitness = local_fitness

        return self.best_solution, self.best_fitness