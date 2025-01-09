import numpy as np

class HybridDE_NM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.F = 0.9  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.scores = np.full(self.population_size, np.inf)
        self.evaluations = 0

    def differential_evolution_step(self, func):
        for i in range(self.population_size):
            if self.evaluations >= self.budget:
                break
            indices = list(range(self.population_size))
            indices.remove(i)
            a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
            mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
            cross_points = np.random.rand(self.dim) < self.CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, self.population[i])
            score = func(trial)
            self.evaluations += 1
            if score < self.scores[i]:
                self.scores[i] = score
                self.population[i] = trial

    def nelder_mead_step(self, func):
        if self.evaluations >= self.budget:
            return
        indices = np.argsort(self.scores)
        best, worst = self.population[indices[0]], self.population[indices[-1]]
        centroid = np.mean(self.population[indices[:-1]], axis=0)
        reflection = np.clip(centroid + (centroid - worst), self.lower_bound, self.upper_bound)
        reflection_score = func(reflection)
        self.evaluations += 1
        if reflection_score < self.scores[indices[-2]]:
            if reflection_score < self.scores[indices[0]]:
                expansion = np.clip(centroid + 2 * (reflection - centroid), self.lower_bound, self.upper_bound)
                expansion_score = func(expansion)
                self.evaluations += 1
                if expansion_score < reflection_score:
                    self.population[indices[-1]] = expansion
                    self.scores[indices[-1]] = expansion_score
                else:
                    self.population[indices[-1]] = reflection
                    self.scores[indices[-1]] = reflection_score
            else:
                self.population[indices[-1]] = reflection
                self.scores[indices[-1]] = reflection_score
        else:
            contraction = np.clip(centroid + 0.5 * (worst - centroid), self.lower_bound, self.upper_bound)
            contraction_score = func(contraction)
            self.evaluations += 1
            if contraction_score < self.scores[indices[-1]]:
                self.population[indices[-1]] = contraction
                self.scores[indices[-1]] = contraction_score
            else:
                for j in range(1, self.population_size):
                    self.population[indices[j]] = best + 0.5 * (self.population[indices[j]] - best)
                    self.scores[indices[j]] = func(self.population[indices[j]])
                    self.evaluations += 1
                    if self.evaluations >= self.budget:
                        break

    def __call__(self, func):
        self.scores = np.array([func(ind) for ind in self.population])
        self.evaluations = self.population_size
        while self.evaluations < self.budget:
            self.differential_evolution_step(func)
            self.nelder_mead_step(func)
        best_index = np.argmin(self.scores)
        return self.population[best_index], self.scores[best_index]