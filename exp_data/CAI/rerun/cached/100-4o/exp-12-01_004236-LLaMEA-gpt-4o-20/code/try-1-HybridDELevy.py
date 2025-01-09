import numpy as np

class HybridDELevy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20  # Population size
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, np.inf)

    def levy_flight(self, beta=1.5):
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / np.abs(v) ** (1 / beta)
        return 0.01 * step

    def adaptive_mutation_rate(self, iteration):
        return max(0.5, 1.0 - iteration / self.budget)

    def __call__(self, func):
        eval_count = 0
        iteration = 0
        # Evaluate initial population
        for i in range(self.pop_size):
            self.fitness[i] = func(self.population[i])
            eval_count += 1
            if eval_count >= self.budget:
                return self.population[np.argmin(self.fitness)]

        while eval_count < self.budget:
            for i in range(self.pop_size):
                # Mutation (Differential Evolution)
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                F_adaptive = self.adaptive_mutation_rate(iteration)
                mutant = self.population[a] + F_adaptive * (self.population[b] - self.population[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                
                # Crossover
                crossover = np.random.rand(self.dim) < self.CR
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial = np.where(crossover, mutant, self.population[i])

                # Adaptive Lévy Flight
                if np.random.rand() < 0.1:
                    trial += self.levy_flight()

                # Evaluation
                trial_fitness = func(trial)
                eval_count += 1

                # Selection
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness

                if eval_count >= self.budget:
                    break
            iteration += 1

        best_index = np.argmin(self.fitness)
        return self.population[best_index]