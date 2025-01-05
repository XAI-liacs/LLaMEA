import numpy as np

class AdaptiveLevyDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.base_pop_size = 5 * dim  # Changed
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.population = np.random.uniform(-5, 5, (self.base_pop_size, dim))  # Changed
        self.fitness = np.full(self.base_pop_size, np.inf)  # Changed
        self.evaluate_population()

    def levy_flight(self, L):
        return 0.01 * (np.random.normal(size=self.dim) / np.power(np.abs(np.random.normal(size=self.dim)), 1/L))
    
    def evaluate_population(self):
        for i in range(self.population.shape[0]):  # Changed
            self.fitness[i] = func(self.population[i])
            if self.fitness[i] < self.f_opt:
                self.f_opt = self.fitness[i]
                self.x_opt = self.population[i].copy()

    def __call__(self, func):
        evaluations = self.population.shape[0]  # Changed
        L = 1.5  # Lévy flight exponent
        
        while evaluations < self.budget:
            # Adaptive population size
            current_pop_size = int(self.base_pop_size * (1 + 0.5 * np.sin(evaluations / self.budget * np.pi)))  # Changed
            self.population = np.resize(self.population, (current_pop_size, self.dim))  # Changed
            self.fitness = np.resize(self.fitness, current_pop_size)  # Changed

            for i in range(current_pop_size):  # Changed
                indices = list(range(current_pop_size))  # Changed
                indices.remove(i)
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                
                mutant_vector = a + self.mutation_factor * (b - c)
                mutant_vector = np.clip(mutant_vector, -5, 5)
                
                trial_vector = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant_vector, self.population[i])
                
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
                
            # Periodic re-initialization
            if evaluations % (self.budget // 10) == 0:  # Changed
                self.population = np.random.uniform(-5, 5, (current_pop_size, self.dim))  # Changed
                self.evaluate_population()  # Changed
                
            self.mutation_factor = 0.5 + (0.5 * (self.budget - evaluations) / self.budget)
            self.crossover_rate = 0.5 + (0.5 * evaluations / self.budget)
        
        return self.f_opt, self.x_opt