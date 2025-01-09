import numpy as np

class ADEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.best_solution = None
        self.best_fitness = np.inf
        self.mutation_factor = 0.5
        self.crossover_probability = 0.7

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.population_size):
                a, b, c = self._select_parents(i)
                mutant = self._mutate(a, b, c)
                trial = self._crossover(self.population[i], mutant)
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness

                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial

                if evaluations >= self.budget:
                    break

            self._adapt_mutation_strategy()

        return self.best_solution

    def _select_parents(self, idx):
        candidates = list(range(self.population_size))
        candidates.remove(idx)
        selected = np.random.choice(candidates, 3, replace=False)
        return self.population[selected[0]], self.population[selected[1]], self.population[selected[2]]

    def _mutate(self, a, b, c):
        mutant = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)
        return mutant

    def _crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_probability
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def _adapt_mutation_strategy(self):
        success_rate = np.mean(self.fitness < np.min(self.fitness) + (np.max(self.fitness) - np.min(self.fitness)) / 5.0)
        if success_rate < 0.2:
            self.mutation_factor *= 0.9
        elif success_rate > 0.5:
            self.mutation_factor *= 1.1
        self.mutation_factor = np.clip(self.mutation_factor, 0.1, 0.9)