import numpy as np

class AdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.population = np.random.uniform(-5, 5, (self.population_size, self.dim))
        self.fitness = np.inf * np.ones(self.population_size)
        self.func_evals = 0
        self.F = 0.5
        self.CR = 0.9

    def _evaluate_population(self, func):
        for i in range(self.population_size):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])
                self.func_evals += 1

    def _mutate(self, target_idx):
        idxs = [idx for idx in range(self.population_size) if idx != target_idx]
        a, b, c = np.random.choice(idxs, 3, replace=False)
        mutant_vector = self.population[a] + self.F * (self.population[b] - self.population[c])
        mutant_vector = np.clip(mutant_vector, -5, 5)
        return mutant_vector

    def _crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.CR
        trial_vector = np.where(crossover_mask, mutant, target)
        return trial_vector

    def _select(self, target_idx, trial_vector, trial_fitness):
        if trial_fitness < self.fitness[target_idx]:
            self.population[target_idx] = trial_vector
            self.fitness[target_idx] = trial_fitness

    def _dynamic_fuzzy_clustering(self):
        centers = self.population[np.argsort(self.fitness)[:self.dim]]
        return centers

    def __call__(self, func):
        self._evaluate_population(func)
        
        while self.func_evals < self.budget:
            for i in range(self.population_size):
                if self.func_evals >= self.budget:
                    break

                mutant = self._mutate(i)
                trial = self._crossover(self.population[i], mutant)
                trial_fitness = func(trial)
                self.func_evals += 1

                self._select(i, trial, trial_fitness)
            
            cluster_centers = self._dynamic_fuzzy_clustering()
            for center in cluster_centers:
                perturbation = np.random.uniform(-0.1, 0.1, self.dim)
                new_point = np.clip(center + perturbation, -5, 5)
                new_fitness = func(new_point)
                self.func_evals += 1

                if new_fitness < np.max(self.fitness):
                    worst_idx = np.argmax(self.fitness)
                    self.population[worst_idx] = new_point
                    self.fitness[worst_idx] = new_fitness

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]