import numpy as np

class MEQES:
    def __init__(self, budget, dim, ensemble_size=5, pop_size=20, quantum_rate=0.2, mutation_rate=0.1, crossover_rate=0.3):
        self.budget = budget
        self.dim = dim
        self.ensemble_size = ensemble_size
        self.pop_size = pop_size
        self.quantum_rate = quantum_rate
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_position = None
        best_value = float('inf')

        ensembles = [self.initialize_population(lb, ub) for _ in range(self.ensemble_size)]
        global_best_value = float('inf')

        while self.evaluations < self.budget:
            for ensemble in ensembles:
                for individual in range(self.pop_size):
                    if np.random.rand() < self.quantum_rate:
                        self.quantum_perturbation(ensemble, individual, lb, ub)

                    value = func(ensemble[individual])
                    self.evaluations += 1

                    if value < best_value:
                        best_value = value
                        best_position = ensemble[individual]

                    if value < global_best_value:
                        global_best_value = value
                        self.adapt_strategy(ensemble, individual)

                    if self.evaluations >= self.budget:
                        break

                self.crossover(ensemble, lb, ub)
                self.mutate(ensemble, lb, ub)

                if self.evaluations >= self.budget:
                    break

        return best_position

    def initialize_population(self, lb, ub):
        return np.random.uniform(lb, ub, (self.pop_size, self.dim))

    def quantum_perturbation(self, ensemble, individual, lb, ub):
        perturbation = (np.random.rand(self.dim) - 0.5) * (ub - lb) * 0.1
        ensemble[individual] = np.clip(ensemble[individual] + perturbation, lb, ub)

    def mutate(self, ensemble, lb, ub):
        for i in range(self.pop_size):
            if np.random.rand() < self.mutation_rate:
                mutation = (np.random.rand(self.dim) - 0.5) * 0.05 * (ub - lb)
                ensemble[i] = np.clip(ensemble[i] + mutation, lb, ub)

    def crossover(self, ensemble, lb, ub):
        for i in range(0, self.pop_size, 2):
            if i + 1 < self.pop_size and np.random.rand() < self.crossover_rate:
                crossover_point = np.random.randint(1, self.dim)
                ensemble[i][:crossover_point], ensemble[i + 1][:crossover_point] = (
                    ensemble[i + 1][:crossover_point].copy(), ensemble[i][:crossover_point].copy())

    def adapt_strategy(self, ensemble, individual):
        # Dynamic adaptation logic based on individual performance
        quantum_adjustment = 0.05 * np.random.rand()
        self.quantum_rate = max(0.05, min(0.3, self.quantum_rate + quantum_adjustment))