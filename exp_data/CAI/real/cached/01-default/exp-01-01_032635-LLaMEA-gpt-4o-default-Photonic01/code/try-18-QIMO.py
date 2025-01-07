import numpy as np

class QIMO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget)
        self.population = None
        self.fitness = None
        self.best_solution = None
        self.best_fitness = np.inf
        self.quantum_amplitude = 0.1
        self.local_search_prob = 0.3
        self.mutation_rate = 0.05

    def initialize_population(self, lb, ub):
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)

    def quantum_search(self, candidate, best):
        direction = np.random.choice([-1, 1], size=self.dim)
        step = self.quantum_amplitude * np.random.random(self.dim)
        new_candidate = candidate + direction * step * (best - candidate)
        return new_candidate

    def memetic_local_search(self, solution, lb, ub):
        perturbation = np.random.normal(0, 0.1, self.dim)
        new_solution = solution + perturbation
        return np.clip(new_solution, lb, ub)

    def mutate(self, candidate, lb, ub):
        mutation_vector = candidate + self.mutation_rate * np.random.normal(0, 1, self.dim)
        return np.clip(mutation_vector, lb, ub)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                # Select a candidate and apply quantum-inspired search
                candidate = self.population[i]
                new_candidate = self.quantum_search(candidate, self.best_solution if self.best_solution is not None else candidate)

                # Apply memetic local search with some probability
                if np.random.rand() < self.local_search_prob:
                    new_candidate = self.memetic_local_search(new_candidate, lb, ub)

                # Evaluate the new candidate
                new_fitness = func(new_candidate)
                evaluations += 1

                # Apply mutation
                if np.random.rand() < self.mutation_rate:
                    new_candidate = self.mutate(new_candidate, lb, ub)
                    new_fitness = func(new_candidate)
                    evaluations += 1

                # Update population and best solution if applicable
                if new_fitness < self.fitness[i]:
                    self.fitness[i] = new_fitness
                    self.population[i] = new_candidate.copy()

                if new_fitness < self.best_fitness:
                    self.best_fitness = new_fitness
                    self.best_solution = new_candidate.copy()

        return self.best_solution, self.best_fitness