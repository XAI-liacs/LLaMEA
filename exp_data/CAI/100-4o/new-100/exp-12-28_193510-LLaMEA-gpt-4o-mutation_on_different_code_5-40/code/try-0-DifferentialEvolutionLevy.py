import numpy as np

class DifferentialEvolutionLevy:
    def __init__(self, budget=10000, dim=10, population_size=50, F=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None
        self.bounds = (-5.0, 5.0)
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)

    def levy_flight(self, u=0, sigma=0.1):
        # Implementing Mantegna's algorithm for Levy flights
        beta = 1.5
        sigma_u = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                   (np.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma_u, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / np.abs(v)**(1 / beta)
        return u + sigma * step

    def mutate(self, idx):
        indices = [i for i in range(self.population_size) if i != idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant_vector = np.clip(self.population[a] + self.F * (self.population[b] - self.population[c]), 
                                self.bounds[0], self.bounds[1])
        return mutant_vector

    def crossover(self, target_vector, mutant_vector):
        crossover_mask = np.random.rand(self.dim) < self.CR
        trial_vector = np.where(crossover_mask, mutant_vector, target_vector)
        return trial_vector

    def __call__(self, func):
        evaluations = 0
        for i in range(self.population_size):
            self.fitness[i] = func(self.population[i])
            evaluations += 1
            if self.fitness[i] < self.f_opt:
                self.f_opt = self.fitness[i]
                self.x_opt = self.population[i]

        while evaluations < self.budget:
            for i in range(self.population_size):
                mutant_vector = self.mutate(i)
                trial_vector = self.crossover(self.population[i], mutant_vector)
                
                # Adaptive Levy Flight for exploitation
                if np.random.rand() < 0.5:
                    levy_step = self.levy_flight()
                    trial_vector = np.clip(trial_vector + levy_step, self.bounds[0], self.bounds[1])
                
                f_trial = func(trial_vector)
                evaluations += 1
                
                if f_trial < self.fitness[i]:
                    self.population[i] = trial_vector
                    self.fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial_vector

                if evaluations >= self.budget:
                    break

        return self.f_opt, self.x_opt