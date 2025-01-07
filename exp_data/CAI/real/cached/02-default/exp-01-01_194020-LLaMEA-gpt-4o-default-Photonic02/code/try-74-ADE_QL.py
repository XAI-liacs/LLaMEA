import numpy as np

class ADE_QL:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7
        self.best_solution = None
        self.best_value = float('inf')

    def initialize_population(self, lb, ub):
        return [lb + (ub - lb) * np.random.rand(self.dim) for _ in range(self.population_size)]

    def mutate(self, target_idx, population, lb, ub):
        indices = [i for i in range(self.population_size) if i != target_idx]
        a, b, c = population[np.random.choice(indices)], population[np.random.choice(indices)], population[np.random.choice(indices)]
        mutated_vector = a + self.mutation_factor * (b - c)
        quantum_leap = np.random.rand(self.dim) < 0.1
        mutated_vector[quantum_leap] = lb[quantum_leap] + (ub[quantum_leap] - lb[quantum_leap]) * np.random.rand(np.sum(quantum_leap))
        return np.clip(mutated_vector, lb, ub)

    def crossover(self, target, mutant):
        crossover_vector = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, target)
        return crossover_vector

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        population = self.initialize_population(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            new_population = []
            for idx, target in enumerate(population):
                mutant = self.mutate(idx, population, lb, ub)
                trial = self.crossover(target, mutant)
                trial_value = func(trial)
                evaluations += 1

                if trial_value < self.best_value:
                    self.best_value = trial_value
                    self.best_solution = trial.copy()

                target_value = func(target)
                if trial_value < target_value:
                    new_population.append(trial)
                else:
                    new_population.append(target)

                if evaluations >= self.budget:
                    break

            population = new_population

        return self.best_solution, self.best_value