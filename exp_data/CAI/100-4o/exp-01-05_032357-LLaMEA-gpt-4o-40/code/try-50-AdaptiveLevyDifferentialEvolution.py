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

    def levy_flight(self, L):
        return 0.01 * (np.random.normal(size=self.dim) / np.power(np.abs(np.random.normal(size=self.dim)), 1/L))
    
    def evaluate_population(self):
        for i in range(self.population_size):
            self.fitness[i] = func(self.population[i])
            if self.fitness[i] < self.f_opt:
                self.f_opt = self.fitness[i]
                self.x_opt = self.population[i].copy()

    def tournament_selection(self, k=3):
        participants = np.random.choice(self.population_size, k, replace=False)
        best_index = participants[np.argmin(self.fitness[participants])]
        return self.population[best_index]

    def __call__(self, func):
        evaluations = self.population_size
        L = 1.5
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                a = self.tournament_selection()
                b, c = np.random.choice(self.population_size, 2, replace=False)
                
                mutant_vector = a + self.mutation_factor * (self.population[b] - self.population[c])
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
                
            self.mutation_factor = 0.5 + (0.5 * (self.budget - evaluations) / self.budget)
            self.crossover_rate = 0.5 + (0.5 * evaluations / self.budget)
            if evaluations % (self.budget // 10) == 0:
                self.population_size = int(self.population_size * 0.9)
                self.population = self.population[:self.population_size]
                self.fitness = self.fitness[:self.population_size]
        
        return self.f_opt, self.x_opt