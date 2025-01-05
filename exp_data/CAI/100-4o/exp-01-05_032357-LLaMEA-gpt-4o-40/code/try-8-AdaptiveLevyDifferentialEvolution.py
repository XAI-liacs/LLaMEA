import numpy as np

class AdaptiveLevyDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 10 * dim
        self.mutation_factor = np.full(self.population_size, 0.8)
        self.crossover_rate = np.full(self.population_size, 0.9)
        self.population = np.random.uniform(-5, 5, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.evaluate_population()
        self.success_count = np.zeros(self.population_size)
        self.failure_count = np.zeros(self.population_size)

    def levy_flight(self, L, scale=0.01):
        return scale * (np.random.normal(size=self.dim) / np.power(np.abs(np.random.normal(size=self.dim)), 1/L))
    
    def evaluate_population(self):
        for i in range(self.population_size):
            self.fitness[i] = func(self.population[i])
            if self.fitness[i] < self.f_opt:
                self.f_opt = self.fitness[i]
                self.x_opt = self.population[i].copy()

    def __call__(self, func):
        evaluations = self.population_size
        L = 1.5
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                
                mutant_vector = a + self.mutation_factor[i] * (b - c)
                mutant_vector = np.clip(mutant_vector, -5, 5)
                
                trial_vector = np.where(np.random.rand(self.dim) < self.crossover_rate[i], mutant_vector, self.population[i])
                
                if np.random.rand() < 0.1:
                    scale = 0.01 + 0.09 * (self.budget - evaluations) / self.budget
                    trial_vector += self.levy_flight(L, scale)
                    trial_vector = np.clip(trial_vector, -5, 5)

                trial_fitness = func(trial_vector)
                evaluations += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial_vector
                    self.fitness[i] = trial_fitness
                    self.success_count[i] += 1
                    self.failure_count[i] = 0
                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial_vector.copy()
                else:
                    self.failure_count[i] += 1
                
                if self.failure_count[i] > 2:
                    self.mutation_factor[i] *= 0.95
                    self.crossover_rate[i] *= 0.95
                elif self.success_count[i] > 2:
                    self.mutation_factor[i] *= 1.05
                    self.crossover_rate[i] *= 1.05

                if evaluations >= self.budget:
                    break

        return self.f_opt, self.x_opt