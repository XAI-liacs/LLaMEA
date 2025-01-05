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

    def levy_flight(self, L):
        u = np.random.normal(size=self.dim) * 0.01
        v = np.random.normal(size=self.dim)
        step = u / np.power(np.abs(v), 1/L)
        return step
    
    def evaluate_population(self):
        for i in range(self.initial_population_size):
            self.fitness[i] = func(self.population[i])
            if self.fitness[i] < self.f_opt:
                self.f_opt = self.fitness[i]
                self.x_opt = self.population[i].copy()

    def dynamic_population_size(self, evaluations):
        return max(5, int(self.initial_population_size * (0.5 + 0.5 * np.cos(np.pi * evaluations / self.budget))))
    
    def __call__(self, func):
        evaluations = self.initial_population_size
        L = 1.5  # Lévy flight exponent
        
        while evaluations < self.budget:
            current_population_size = self.dynamic_population_size(evaluations)
            for i in range(current_population_size):
                indices = list(range(current_population_size))
                indices.remove(i)
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                
                mutant_vector = a + 0.8 * (b - c)
                mutant_vector = np.clip(mutant_vector, -5, 5)
                
                trial_vector = np.where(np.random.rand(self.dim) < 0.9, mutant_vector, self.population[i])
                
                if np.random.rand() < 0.1:
                    trial_vector += self.levy_flight(L)
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
                
        return self.f_opt, self.x_opt