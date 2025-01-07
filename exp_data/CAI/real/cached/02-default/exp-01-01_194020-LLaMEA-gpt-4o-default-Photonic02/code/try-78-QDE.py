import numpy as np

class QDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')
        self.population_size = 20
        self.population = []

    def initialize_population(self, lb, ub):
        population = []
        for _ in range(self.population_size):
            position = lb + (ub - lb) * np.random.rand(self.dim)
            population.append({'position': position, 'value': float('inf')})
        return population

    def mutate(self, target_index, lb, ub):
        indices = [i for i in range(self.population_size) if i != target_index]
        a, b, c = np.random.choice(indices, 3, replace=False)
        F = 0.8  # mutation factor
        donor_vector = self.population[a]['position'] + F * (self.population[b]['position'] - self.population[c]['position'])
        return np.clip(donor_vector, lb, ub)

    def crossover(self, target_vector, donor_vector):
        CR = 0.9  # crossover probability
        trial_vector = np.where(np.random.rand(self.dim) < CR, donor_vector, target_vector)
        return trial_vector

    def quantum_influence(self, trial_vector, global_best, lb, ub, beta):
        direction = np.sign(np.random.rand(self.dim) - 0.5)
        phi = np.arccos(1 - 2 * np.random.rand(self.dim))
        influenced_vector = trial_vector + beta * np.abs(global_best - trial_vector) * np.tan(phi) * direction
        return np.clip(influenced_vector, lb, ub)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        self.population = self.initialize_population(lb, ub)
        global_best = None
        global_best_value = float('inf')

        while evaluations < self.budget:
            for i, individual in enumerate(self.population):
                target_vector = individual['position']
                donor_vector = self.mutate(i, lb, ub)
                trial_vector = self.crossover(target_vector, donor_vector)

                trial_value = func(trial_vector)
                evaluations += 1

                if trial_value < individual['value']:
                    individual['position'] = trial_vector
                    individual['value'] = trial_value

                if trial_value < global_best_value:
                    global_best_value = trial_value
                    global_best = trial_vector.copy()

                if evaluations >= self.budget:
                    break

            beta = 1.0 - evaluations / self.budget
            for individual in self.population:
                individual['position'] = self.quantum_influence(individual['position'], global_best, lb, ub, beta)

        return global_best, global_best_value