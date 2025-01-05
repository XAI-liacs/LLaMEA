import numpy as np

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 2.05
        self.c2 = 2.05
        self.w = 0.729  # Inertia weight
        self.f = 0.8  # Differential evolution scale factor
        self.cr = 0.9  # Crossover rate
        self.mutation_prob = 0.1  # Probability for adaptive mutation
        self.position = None
        self.velocity = None
        self.pbest = None
        self.pbest_scores = None
        self.gbest = None
        self.gbest_score = float('inf')

    def initialize(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        self.position = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.velocity = np.random.rand(self.population_size, self.dim) - 0.5
        self.pbest = np.copy(self.position)
        self.pbest_scores = np.full(self.population_size, float('inf'))

    def evaluate(self, func):
        scores = np.array([func(p) for p in self.position])
        for i in range(self.population_size):
            if scores[i] < self.pbest_scores[i]:
                self.pbest_scores[i] = scores[i]
                self.pbest[i] = self.position[i]
            if scores[i] < self.gbest_score:
                self.gbest_score = scores[i]
                self.gbest = self.position[i]
        return scores

    def update_velocity_position(self):
        r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
        cognitive = self.c1 * r1 * (self.pbest - self.position)
        social = self.c2 * r2 * (self.gbest - self.position)
        self.velocity = self.w * self.velocity + cognitive + social
        self.position += self.velocity

    def differential_evolution_mutation(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        new_population = np.copy(self.position)
        for i in range(self.population_size):
            indices = list(range(self.population_size))
            indices.remove(i)
            a, b, c = np.random.choice(indices, 3, replace=False)
            mutant = self.pbest[a] + self.f * (self.pbest[b] - self.pbest[c])
            mutant = np.clip(mutant, lb, ub)
            crossover = np.random.rand(self.dim) < self.cr
            if not np.any(crossover):
                crossover[np.random.randint(0, self.dim)] = True
            new_population[i] = np.where(crossover, mutant, self.position[i])
        return new_population

    def adaptive_mutation(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        for i in range(self.population_size):
            if np.random.rand() < self.mutation_prob:
                mutation_vector = np.random.normal(0, 0.1, self.dim)
                self.position[i] = np.clip(self.position[i] + mutation_vector, lb, ub)

    def __call__(self, func):
        func_calls = 0
        self.initialize(func.bounds)
        while func_calls < self.budget:
            scores = self.evaluate(func)
            func_calls += self.population_size
            self.update_velocity_position()
            new_population = self.differential_evolution_mutation(func.bounds)
            self.position = np.where(
                scores[:, np.newaxis] <= np.array([func(p) for p in new_population])[:, np.newaxis],
                self.position, new_population
            )
            func_calls += self.population_size
            self.adaptive_mutation(func.bounds)
        return self.gbest, self.gbest_score