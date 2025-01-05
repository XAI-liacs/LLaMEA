import numpy as np

class AQDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.best_solution = None
        self.best_value = float('inf')
        self.population = []
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9

    def initialize_population(self, lb, ub):
        population = []
        for _ in range(self.population_size):
            position = lb + (ub - lb) * np.random.rand(self.dim)
            population.append({'position': position, 'value': float('inf')})
        return population

    def differential_mutation(self, target_idx, lb, ub):
        indices = list(range(self.population_size))
        indices.remove(target_idx)
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant_vector = self.population[a]['position'] + self.mutation_factor * (self.population[b]['position'] - self.population[c]['position'])
        mutant_vector = np.clip(mutant_vector, lb, ub)
        return mutant_vector

    def crossover(self, target_vector, mutant_vector, lb, ub):
        crossover_vector = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant_vector, target_vector)
        phi = np.arccos(1 - 2 * np.random.rand(self.dim))
        direction = np.sign(np.random.rand(self.dim) - 0.5)
        quantum_adjustment = crossover_vector + np.tan(phi) * direction * (ub - lb) * 0.1
        quantum_adjustment = np.clip(quantum_adjustment, lb, ub)
        return quantum_adjustment

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        self.population = self.initialize_population(lb, ub)

        while evaluations < self.budget:
            for target_idx, target in enumerate(self.population):
                mutant_vector = self.differential_mutation(target_idx, lb, ub)
                trial_vector = self.crossover(target['position'], mutant_vector, lb, ub)

                trial_value = func(trial_vector)
                evaluations += 1

                if trial_value < target['value']:
                    target['position'], target['value'] = trial_vector, trial_value

                if trial_value < self.best_value:
                    self.best_solution, self.best_value = trial_vector.copy(), trial_value

                if evaluations >= self.budget:
                    break

            self.mutation_factor = 0.5 + 0.5 * (1 - evaluations / self.budget)

        return self.best_solution, self.best_value