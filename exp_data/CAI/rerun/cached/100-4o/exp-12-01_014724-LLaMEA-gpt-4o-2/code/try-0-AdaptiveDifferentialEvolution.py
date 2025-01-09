import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.strategy = 'best1bin'
        self.f = 0.8
        self.cr = 0.9
        self.func_evals = 0

    def _initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def _mutate(self, population, best_idx):
        idxs = np.arange(self.population_size)
        for i in range(self.population_size):
            if self.func_evals >= self.budget:
                break
            a, b, c = population[np.random.choice(idxs[idxs != i], 3, replace=False)]
            if self.strategy == 'best1bin':
                mutant = np.clip(population[best_idx] + self.f * (b - c), self.lower_bound, self.upper_bound)
            else:
                mutant = np.clip(a + self.f * (b - c), self.lower_bound, self.upper_bound)
            yield i, mutant

    def _crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.cr
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def _select(self, population, scores, func, trials):
        for i, trial in trials:
            if self.func_evals >= self.budget:
                break
            trial_score = func(trial)
            self.func_evals += 1
            if trial_score < scores[i]:
                population[i] = trial
                scores[i] = trial_score
        return population, scores

    def __call__(self, func):
        population = self._initialize_population()
        scores = np.array([func(ind) for ind in population])
        self.func_evals += self.population_size
        best_idx = np.argmin(scores)
        best = population[best_idx]
        best_score = scores[best_idx]
        
        while self.func_evals < self.budget:
            trials = list(self._mutate(population, best_idx))
            trials = [(i, self._crossover(population[i], mutant)) for i, mutant in trials]
            population, scores = self._select(population, scores, func, trials)
            best_idx = np.argmin(scores)
            if scores[best_idx] < best_score:
                best = population[best_idx]
                best_score = scores[best_idx]
                
        return best