import numpy as np

class EnhancedDynamicSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 50
        self.min_population_size = 10
        self.c1 = 2.05
        self.c2 = 2.05
        self.w_max = 0.9  # Maximum inertia weight
        self.w_min = 0.4  # Minimum inertia weight
        self.f = 0.8  # Differential evolution scale factor
        self.cr = 0.9  # Crossover rate
        self.position = None
        self.velocity = None
        self.pbest = None
        self.pbest_scores = None
        self.gbest = None
        self.gbest_score = float('inf')
        self.chaotic_sequence = self.generate_chaotic_sequence()
        self.iteration = 0
        self.func_calls = 0

    def generate_chaotic_sequence(self, length=10000):
        x = 0.7
        sequence = []
        for _ in range(length):
            x = 4 * x * (1 - x)  # Logistic map for chaos
            sequence.append(x)
        return np.array(sequence)
    
    def chaotic_parameter(self, index):
        return self.chaotic_sequence[index % len(self.chaotic_sequence)]
    
    def initialize(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        self.position = lb + (ub - lb) * np.random.rand(self.initial_population_size, self.dim)
        self.velocity = np.random.rand(self.initial_population_size, self.dim) - 0.5
        self.pbest = np.copy(self.position)
        self.pbest_scores = np.full(self.initial_population_size, float('inf'))
        self.population_size = self.initial_population_size

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

    def adaptive_inertia_weight(self):
        return self.w_max - ((self.w_max - self.w_min) * self.iteration / (self.budget // self.initial_population_size))

    def update_velocity_position(self):
        r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
        chaotic_factor = self.chaotic_parameter(self.iteration)
        cognitive = chaotic_factor * self.c1 * r1 * (self.pbest - self.position)
        social = chaotic_factor * self.c2 * r2 * (self.gbest - self.position)
        w = self.adaptive_inertia_weight()
        self.velocity = w * self.velocity + cognitive + social
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

    def adapt_population_size(self):
        if self.iteration % 10 == 0 and self.population_size > self.min_population_size:
            self.population_size = max(self.min_population_size, self.population_size - 5)
            self.position = self.position[:self.population_size]
            self.velocity = self.velocity[:self.population_size]
            self.pbest = self.pbest[:self.population_size]
            self.pbest_scores = self.pbest_scores[:self.population_size]

    def __call__(self, func):
        self.initialize(func.bounds)
        while self.func_calls < self.budget:
            scores = self.evaluate(func)
            self.func_calls += self.population_size
            self.update_velocity_position()
            new_population = self.differential_evolution_mutation(func.bounds)
            self.position = np.where(
                scores[:, np.newaxis] <= np.array([func(p) for p in new_population])[:, np.newaxis],
                self.position, new_population
            )
            self.func_calls += self.population_size
            self.adapt_population_size()
            self.iteration += 1
        return self.gbest, self.gbest_score