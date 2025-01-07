import numpy as np

class QuantumDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, min(50, budget // 10))
        self.population = None
        self.fitness = None
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.scale_factor = 0.5
        self.crossover_prob = 0.7
        self.adaptive_rate = 0.1

    def initialize_population(self, lb, ub):
        self.population = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.fitness = np.full(self.population_size, float('inf'))

    def evaluate_population(self, func):
        for i in range(self.population_size):
            fitness_value = func(self.population[i])
            if fitness_value < self.fitness[i]:
                self.fitness[i] = fitness_value
                if fitness_value < self.global_best_fitness:
                    self.global_best_fitness = fitness_value
                    self.global_best_position = np.copy(self.population[i])

    def mutate_and_crossover(self, lb, ub):
        new_population = np.copy(self.population)
        for i in range(self.population_size):
            indices = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
            mutant_vector = a + self.scale_factor * (b - c)
            trial_vector = np.copy(self.population[i])
            
            jrand = np.random.randint(self.dim)
            for j in range(self.dim):
                if np.random.rand() < self.crossover_prob or j == jrand:
                    trial_vector[j] = mutant_vector[j]
            
            trial_fitness = func(trial_vector)
            if trial_fitness < self.fitness[i]:
                new_population[i] = trial_vector
                self.fitness[i] = trial_fitness
                if trial_fitness < self.global_best_fitness:
                    self.global_best_fitness = trial_fitness
                    self.global_best_position = np.copy(trial_vector)
        
        self.population = np.clip(new_population, lb, ub)

    def apply_quantum_swarming(self, lb, ub):
        for i in range(self.population_size):
            if np.random.rand() < self.adaptive_rate:
                random_vector = lb + (ub - lb) * np.random.rand(self.dim)
                self.population[i] = np.mean([self.population[i], random_vector], axis=0)
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
            self.apply_quantum_swarming(lb, ub)

        return self.global_best_position, self.global_best_fitness