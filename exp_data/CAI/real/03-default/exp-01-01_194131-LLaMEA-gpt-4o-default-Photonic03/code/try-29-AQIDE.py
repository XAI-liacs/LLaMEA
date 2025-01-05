import numpy as np

class AQIDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.positions = np.random.rand(self.population_size, dim)
        self.fitness = np.full(self.population_size, float('inf'))
        self.best = None
        self.best_score = float('inf')
        self.mutation_strategies = [self._rand_1, self._best_1, self._current_to_best_1]
        self.strategy_probabilities = np.array([1/3, 1/3, 1/3])
        self.evaluations = 0

    def _rand_1(self, a, b, c, d, F=0.8):
        return a + F * (b - c)

    def _best_1(self, a, b, c, best, F=0.8):
        return best + F * (b - c)

    def _current_to_best_1(self, a, b, c, best, F=0.8):
        return a + F * (best - a) + F * (b - c)

    def _select_parents(self, idx):
        indices = list(range(self.population_size))
        indices.remove(idx)
        selected = np.random.choice(indices, size=3, replace=False)
        return selected

    def _mutate(self, idx):
        a, b, c = self._select_parents(idx)
        best_idx = np.argmin(self.fitness)
        strategy_idx = np.random.choice(len(self.mutation_strategies), p=self.strategy_probabilities)
        return self.mutation_strategies[strategy_idx](self.positions[a], self.positions[b], self.positions[c], self.positions[best_idx])

    def _crossover(self, target, donor, CR=0.9):
        mask = np.random.rand(self.dim) < CR
        trial = np.where(mask, donor, target)
        return trial

    def _update_strategy_probabilities(self):
        success_rates = self.strategy_successes / (self.strategy_tries + 1e-6)
        self.strategy_probabilities = success_rates / success_rates.sum()

    def _adapt_search_space(self, factor=0.9):
        if self.evaluations % (self.budget // 5) == 0:
            self.positions = self.best + factor * (self.positions - self.best)
            self.positions = np.clip(self.positions, self.bounds.lb, self.bounds.ub)

    def __call__(self, func):
        self.bounds = func.bounds
        self.positions = self.bounds.lb + (self.bounds.ub - self.bounds.lb) * np.random.rand(self.population_size, self.dim)
        self.strategy_successes = np.zeros(len(self.mutation_strategies))
        self.strategy_tries = np.zeros(len(self.mutation_strategies))

        for i in range(self.population_size):
            score = func(self.positions[i])
            self.fitness[i] = score
            if score < self.best_score:
                self.best = self.positions[i].copy()
                self.best_score = score
            self.evaluations += 1
            if self.evaluations >= self.budget:
                return self.best

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                donor = self._mutate(i)
                trial = self._crossover(self.positions[i], donor)
                trial = np.clip(trial, self.bounds.lb, self.bounds.ub)
                trial_score = func(trial)
                self.evaluations += 1

                if trial_score < self.fitness[i]:
                    self.positions[i] = trial
                    self.fitness[i] = trial_score
                    self.strategy_successes[strategy_idx] += 1
                    if trial_score < self.best_score:
                        self.best = trial
                        self.best_score = trial_score

                self.strategy_tries[strategy_idx] += 1
                if self.evaluations >= self.budget:
                    break

            self._update_strategy_probabilities()
            self._adapt_search_space()

        return self.best