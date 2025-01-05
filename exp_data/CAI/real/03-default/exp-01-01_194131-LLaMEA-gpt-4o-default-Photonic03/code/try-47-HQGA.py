import numpy as np

class HQGA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.chromosomes = np.random.rand(self.population_size, dim)
        self.scores = np.full(self.population_size, float('inf'))
        self.best_chromosome = None
        self.best_score = float('inf')
        self.evaluations = 0
        self.crossover_rate = 0.8
        self.mutation_rate = 0.1
        self.alpha = 0.9  # Adaptive parameter for mutation rate

    def quantum_encoding(self):
        return np.cos(np.pi * self.chromosomes), np.sin(np.pi * self.chromosomes)

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, self.dim - 1)
            return np.concatenate((parent1[:point], parent2[point:]))
        return parent1 if np.random.rand() > 0.5 else parent2

    def mutate(self, chromosome):
        for i in range(self.dim):
            if np.random.rand() < self.mutation_rate:
                chromosome[i] = np.random.rand()
        return chromosome

    def adapt_parameters(self):
        if self.evaluations % (self.budget // 5) == 0:
            self.mutation_rate = max(0.05, self.alpha * self.mutation_rate)

    def __call__(self, func):
        # Initialize population
        bounds_range = func.bounds.ub - func.bounds.lb
        self.chromosomes = func.bounds.lb + bounds_range * self.chromosomes

        for i in range(self.population_size):
            score = func(self.chromosomes[i])
            self.scores[i] = score
            if score < self.best_score:
                self.best_chromosome = self.chromosomes[i]
                self.best_score = score
            self.evaluations += 1
            if self.evaluations >= self.budget:
                return self.best_chromosome

        # Main optimization loop
        while self.evaluations < self.budget:
            new_population = []
            for _ in range(self.population_size // 2):
                parents = self.chromosomes[np.random.choice(self.population_size, 2, replace=False)]
                offspring1 = self.crossover(parents[0], parents[1])
                offspring2 = self.crossover(parents[1], parents[0])
                new_population.extend([self.mutate(offspring1), self.mutate(offspring2)])
            
            new_population = np.clip(new_population, func.bounds.lb, func.bounds.ub)
            
            for i in range(self.population_size):
                score = func(new_population[i])
                if score < self.scores[i]:
                    self.chromosomes[i] = new_population[i]
                    self.scores[i] = score
                    if score < self.best_score:
                        self.best_chromosome = new_population[i]
                        self.best_score = score
                self.evaluations += 1
                if self.evaluations >= self.budget:
                    return self.best_chromosome

            self.adapt_parameters()

        return self.best_chromosome