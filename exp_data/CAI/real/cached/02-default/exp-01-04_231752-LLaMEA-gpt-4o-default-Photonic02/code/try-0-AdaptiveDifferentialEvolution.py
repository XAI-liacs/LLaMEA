import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + self.dim
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.population = None
        self.scores = None

    def _initialize_population(self, lb, ub):
        self.population = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        self.scores = np.full(self.population_size, np.inf)

    def _select_parents(self, idx):
        indices = list(range(self.population_size))
        indices.remove(idx)
        return np.random.choice(indices, 3, replace=False)

    def _mutate(self, idx):
        a, b, c = self._select_parents(idx)
        mutant_vector = self.population[a] + self.mutation_factor * (self.population[b] - self.population[c])
        return np.clip(mutant_vector, self.lb, self.ub)

    def _crossover(self, target_vector, mutant_vector):
        crossover_mask = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        trial_vector = np.where(crossover_mask, mutant_vector, target_vector)
        return trial_vector

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self._initialize_population(self.lb, self.ub)
        
        eval_count = 0
        while eval_count < self.budget:
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break
                
                target_vector = self.population[i]
                mutant_vector = self._mutate(i)
                trial_vector = self._crossover(target_vector, mutant_vector)
                
                trial_score = func(trial_vector)
                eval_count += 1

                if trial_score < self.scores[i]:
                    self.population[i] = trial_vector
                    self.scores[i] = trial_score

        best_idx = np.argmin(self.scores)
        return self.population[best_idx], self.scores[best_idx]