import numpy as np

class HybridDEAS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.temperature = 100.0
        self.cooling_rate = 0.99
        self.best_solution = None
        self.best_obj = float('inf')

    def initialize_population(self, bounds):
        self.lb, self.ub = bounds.lb, bounds.ub
        self.population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        self.objectives = np.array([float('inf')] * self.population_size)

    def differential_evolution_step(self):
        new_population = np.copy(self.population)
        for i in range(self.population_size):
            indices = np.random.choice(self.population_size, 3, replace=False)
            a, b, c = self.population[indices]
            mutant = a + self.mutation_factor * (b - c)
            mutant = np.clip(mutant, self.lb, self.ub)
            trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, self.population[i])
            trial_obj = func(trial)
            if trial_obj < self.objectives[i]:
                new_population[i] = trial
                self.objectives[i] = trial_obj
        self.population = new_population

    def simulated_annealing_step(self, func):
        for i in range(self.population_size):
            candidate = self.population[i] + np.random.normal(0, 0.1, self.dim)
            candidate = np.clip(candidate, self.lb, self.ub)
            candidate_obj = func(candidate)
            if candidate_obj < self.objectives[i] or np.random.rand() < np.exp((self.objectives[i] - candidate_obj) / self.temperature):
                self.population[i] = candidate
                self.objectives[i] = candidate_obj
        self.temperature *= self.cooling_rate

    def __call__(self, func):
        self.initialize_population(func.bounds)
        evaluations = 0
        
        while evaluations < self.budget:
            self.differential_evolution_step()
            evaluations += self.population_size

            self.simulated_annealing_step(func)
            evaluations += self.population_size

            for i in range(self.population_size):
                if self.objectives[i] < self.best_obj:
                    self.best_obj = self.objectives[i]
                    self.best_solution = self.population[i]
        
        return self.best_solution