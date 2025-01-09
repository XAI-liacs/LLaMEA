import numpy as np

class AdaptiveLevyDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 10 * dim
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.population = np.random.uniform(-5, 5, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.evaluate_population()

    def levy_flight(self, L, scale=0.01):
        return scale * (np.random.normal(size=self.dim) / np.power(np.abs(np.random.normal(size=self.dim)), 1/L))
    
    def gaussian_perturbation(self, scale=0.01):
        return scale * np.random.normal(size=self.dim)

    def evaluate_population(self):
        for i in range(self.population_size):
            self.fitness[i] = func(self.population[i])
            if self.fitness[i] < self.f_opt:
                self.f_opt = self.fitness[i]
                self.x_opt = self.population[i].copy()

    def __call__(self, func):
        evaluations = self.population_size
        L = 1.5  # Lévy flight exponent
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                
                mutant_vector = a + self.mutation_factor * (b - c)
                mutant_vector = np.clip(mutant_vector, -5, 5)
                
                trial_vector = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant_vector, self.population[i])
                
                adaptive_scale = (self.budget - evaluations) / self.budget
                if np.random.rand() < 0.2:  # Increased probability of perturbation
                    if np.random.rand() < 0.5:
                        trial_vector += self.levy_flight(L, adaptive_scale)
                    else:
                        trial_vector += self.gaussian_perturbation(adaptive_scale)
                    trial_vector = np.clip(trial_vector, -5, 5)

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
            self.crossover_rate = 0.6 + (0.4 * evaluations / self.budget)  # Adjusted crossover rate
        
        return self.f_opt, self.x_opt