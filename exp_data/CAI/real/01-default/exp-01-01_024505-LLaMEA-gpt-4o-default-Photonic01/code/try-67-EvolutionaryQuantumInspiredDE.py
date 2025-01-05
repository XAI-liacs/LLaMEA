import numpy as np

class EvolutionaryQuantumInspiredDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, min(50, budget // 10))
        self.population = None
        self.fitness = np.full(self.population_size, float('inf'))
        self.best_solution = None
        self.best_fitness = float('inf')
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.interference_prob = 0.1

    def initialize_population(self, lb, ub):
        self.population = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)

    def evaluate_population(self, func):
        self.fitness = np.array([func(ind) for ind in self.population])
        for i, f in enumerate(self.fitness):
            if f < self.best_fitness:
                self.best_fitness = f
                self.best_solution = self.population[i]
        return self.fitness

    def mutate_and_crossover(self, lb, ub):
        new_population = np.copy(self.population)
        for i in range(self.population_size):
            indices = np.random.choice(self.population_size, 3, replace=False)
            a, b, c = self.population[indices]
            mutant = a + self.mutation_factor * (b - c)
            mutant = np.clip(mutant, lb, ub)

            crossover = np.random.rand(self.dim) < self.crossover_rate
            if not np.any(crossover):
                crossover[np.random.randint(0, self.dim)] = True

            new_population[i] = np.where(crossover, mutant, self.population[i])

        self.population = np.copy(new_population)

    def apply_quantum_interference(self, lb, ub):
        for i in range(self.population_size):
            if np.random.rand() < self.interference_prob:
                interference_vector = lb + (ub - lb) * np.random.rand(self.dim)
                self.population[i] = np.mean([self.population[i], interference_vector], axis=0)
                self.population[i] = np.clip(self.population[i], lb, ub)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            self.evaluate_population(func)
            evaluations += self.population_size

            if evaluations >= self.budget:
                break

            self.mutate_and_crossover(lb, ub)
            self.apply_quantum_interference(lb, ub)

        return self.best_solution, self.best_fitness