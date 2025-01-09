import numpy as np

class ASDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(5, 4 * self.dim)  # Heuristic for population size
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        self.fitness = np.full(self.population_size, np.inf)
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.evaluations = 0

    def adapt_parameters(self):
        mean_fitness = np.mean(self.fitness)
        if mean_fitness < np.percentile(self.fitness, 25):
            self.F = np.clip(self.F + 0.1 * (0.75 - self.F), 0.4, 1.0)
            self.CR = np.clip(self.CR + 0.1 * (0.95 - self.CR), 0.6, 1.0)
        elif mean_fitness > np.percentile(self.fitness, 75):
            self.F = np.clip(self.F - 0.1 * (self.F - 0.2), 0.1, 0.9)
            self.CR = np.clip(self.CR - 0.1 * (self.CR - 0.3), 0.1, 0.7)

    def __call__(self, func):
        self.fitness = np.array([func(ind) for ind in self.population])
        self.evaluations += self.population_size

        while self.evaluations < self.budget:
            self.adapt_parameters()
            new_population = np.copy(self.population)
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = np.clip(self.population[a] + self.F * (self.population[b] - self.population[c]), self.lower_bound, self.upper_bound)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.population[i])
                f_trial = func(trial)
                self.evaluations += 1
                if f_trial < self.fitness[i]:
                    new_population[i], self.fitness[i] = trial, f_trial
            
            self.population = new_population
            if self.evaluations >= self.budget:
                break

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]