import numpy as np

class EnhancedAdaptiveLevyDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.initial_population_size = 10 * dim
        self.population = np.random.uniform(-5, 5, (self.initial_population_size, dim))
        self.fitness = np.full(self.initial_population_size, np.inf)
        self.evaluate_population()
        self.dynamic_population_size = self.initial_population_size

    def levy_flight(self, L):
        return 0.01 * (np.random.normal(size=self.dim) / np.power(np.abs(np.random.normal(size=self.dim) + 1e-8), 1/L))
    
    def evaluate_population(self):
        for i in range(self.dynamic_population_size):
            self.fitness[i] = func(self.population[i])
            if self.fitness[i] < self.f_opt:
                self.f_opt = self.fitness[i]
                self.x_opt = self.population[i].copy()

    def local_search(self, vector):
        perturbation = np.random.normal(0, 0.1, self.dim)
        return np.clip(vector + perturbation, -5, 5)

    def __call__(self, func):
        evaluations = self.dynamic_population_size
        L = 1.5
        
        while evaluations < self.budget:
            for i in range(self.dynamic_population_size):
                indices = list(range(self.dynamic_population_size))
                indices.remove(i)
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                
                mutant_vector = a + self.mutation_factor * (b - c)
                mutant_vector = np.clip(mutant_vector, -5, 5)
                
                trial_vector = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant_vector, self.population[i])
                
                if np.random.rand() < 0.1:
                    trial_vector += self.levy_flight(L)
                    trial_vector = self.local_search(trial_vector)

                trial_fitness = func(trial_vector)
                evaluations += 1
                
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial_vector
                    self.fitness[i] = trial_fitness
                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial_vector.copy()
                
                if evaluations >= self.budget:
                    break
            
            self.mutation_factor = 0.5 + (0.5 * (self.budget - evaluations) / self.budget)
            self.crossover_rate = 0.5 + (0.5 * evaluations / self.budget)
            self.dynamic_population_size = self.initial_population_size - int(evaluations / self.budget * self.initial_population_size) + 1
        
        return self.f_opt, self.x_opt