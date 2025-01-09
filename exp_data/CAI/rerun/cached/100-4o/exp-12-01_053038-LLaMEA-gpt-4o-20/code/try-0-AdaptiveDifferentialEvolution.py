import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.CR = 0.9  # Crossover probability
        self.F = 0.8   # Differential weight
        self.evaluations = 0

    def _evaluate_population(self, func):
        for i in range(self.population_size):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.pop[i])
                self.evaluations += 1
                if self.evaluations >= self.budget:
                    return

    def _select_parents(self):
        indices = np.random.choice(self.population_size, 3, replace=False)
        return indices

    def _mutate(self, base, a, b, c):
        mutant = base + self.F * (a - b + c - base)
        mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
        return mutant

    def _crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def __call__(self, func):
        self._evaluate_population(func)

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                indices = self._select_parents()
                while i in indices:
                    indices = self._select_parents()
                base, a, b = self.pop[indices]

                mutant = self._mutate(base, a, b, self.pop[i])
                trial = self._crossover(self.pop[i], mutant)

                trial_fitness = func(trial)
                self.evaluations += 1

                if trial_fitness < self.fitness[i]:
                    self.pop[i] = trial
                    self.fitness[i] = trial_fitness

            # Adapt F and CR based on diversity
            population_std = np.std(self.pop, axis=0)
            diversity = np.mean(population_std)
            self.F = 0.5 + 0.3 * (1 - diversity) # More diversity, less greediness
            self.CR = 0.5 + 0.4 * (diversity)    # More diversity, more exploration

        best_idx = np.argmin(self.fitness)
        return self.pop[best_idx], self.fitness[best_idx]