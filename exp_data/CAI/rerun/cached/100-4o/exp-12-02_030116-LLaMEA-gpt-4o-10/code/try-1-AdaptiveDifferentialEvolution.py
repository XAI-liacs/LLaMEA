import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 20 + dim * 5  # Population size
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.history = []

    def _initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))

    def _mutate(self, idx, population):
        indices = [i for i in range(self.pop_size) if i != idx]
        a, b, c = population[np.random.choice(indices, 3, replace=False)]
        mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
        return mutant

    def _crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.CR
        return np.where(crossover_mask, mutant, target)

    def _select(self, target, trial, func):
        if func(trial) < func(target):
            return trial
        return target

    def __call__(self, func):
        population = self._initialize_population()
        fitness = np.apply_along_axis(func, 1, population)
        eval_count = self.pop_size

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        while eval_count < self.budget:
            for i in range(self.pop_size):
                mutant = self._mutate(i, population)
                trial = self._crossover(population[i], mutant)
                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                if eval_count >= self.budget:
                    break

            # Adapt F and CR based on past success (optional improvement)
            self.history.append(best_fitness)

        return best_solution