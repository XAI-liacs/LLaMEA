import numpy as np

class HybridOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.cooling_rate = 0.99
        self.initial_temperature = 100
        self.temperature = self.initial_temperature

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def differential_evolution(self, population, func):
        F = 0.8
        CR = 0.9
        new_population = np.copy(population)
        for i in range(self.population_size):
            a, b, c = population[np.random.choice(self.population_size, 3, replace=False)]
            mutant_vector = a + F * (b - c)
            trial_vector = np.where(np.random.rand(self.dim) < CR, mutant_vector, population[i])
            trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)
            if func(trial_vector) < func(population[i]):
                new_population[i] = trial_vector
        return new_population

    def simulated_annealing(self, candidate, func):
        new_candidate = candidate + np.random.normal(0, 1, self.dim)
        new_candidate = np.clip(new_candidate, self.lower_bound, self.upper_bound)
        if np.random.rand() < np.exp((func(candidate) - func(new_candidate)) / self.temperature):
            return new_candidate
        return candidate

    def __call__(self, func):
        population = self.initialize_population()
        best_solution = None
        best_score = float('inf')
        evaluations = 0

        while evaluations < self.budget:
            population = self.differential_evolution(population, func)
            for i in range(self.population_size):
                candidate = population[i]
                candidate = self.simulated_annealing(candidate, func)
                population[i] = candidate
                score = func(candidate)
                evaluations += 1
                if score < best_score:
                    best_score = score
                    best_solution = candidate
                if evaluations >= self.budget:
                    break
            self.temperature *= self.cooling_rate

        return best_solution, best_score