import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.population_size = int(np.clip(20, 5, budget // dim))
        self.pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.scores = np.full(self.population_size, float('inf'))
        self.current_evaluations = 0

    def mutate(self, idx):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = self.pop[a] + self.F * (self.pop[b] - self.pop[c])
        return np.clip(mutant, self.lower_bound, self.upper_bound)

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        offspring = np.where(cross_points, mutant, target)
        return offspring

    def adapt_population_size(self):
        # Adjust population size dynamically during iterations
        factor = np.exp(-self.current_evaluations / self.budget)
        self.population_size = max(5, int(self.population_size * factor))
        self.pop = self.pop[:self.population_size]
        self.scores = self.scores[:self.population_size]

    def __call__(self, func):
        # Initial evaluation
        for i in range(self.population_size):
            self.scores[i] = func(self.pop[i])
            self.current_evaluations += 1
            if self.current_evaluations >= self.budget:
                return self.pop[np.argmin(self.scores)]

        while self.current_evaluations < self.budget:
            for i in range(self.population_size):
                mutant = self.mutate(i)
                offspring = self.crossover(self.pop[i], mutant)
                score = func(offspring)
                self.current_evaluations += 1

                if score < self.scores[i]:
                    self.pop[i] = offspring
                    self.scores[i] = score

                if self.current_evaluations >= self.budget:
                    return self.pop[np.argmin(self.scores)]
            
            # Adapt the population size over time
            self.adapt_population_size()

        return self.pop[np.argmin(self.scores)]