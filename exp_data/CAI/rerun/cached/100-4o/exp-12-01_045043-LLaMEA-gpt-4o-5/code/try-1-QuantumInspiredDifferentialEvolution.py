import numpy as np

class QuantumInspiredDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.best_solution = None
        self.best_fitness = float('inf')
        self.fitness = np.full(self.population_size, float('inf'))
        self.evaluations = 0
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.qbits = np.random.uniform(0, 2 * np.pi, (self.population_size, dim))  # Quantum bits

    def __call__(self, func):
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                # Evaluate fitness and update best solution
                if self.fitness[i] == float('inf'):
                    self.fitness[i] = func(self.population[i])
                    self.evaluations += 1

                if self.fitness[i] < self.best_fitness:
                    self.best_fitness = self.fitness[i]
                    self.best_solution = self.population[i].copy()

                # Mutation and crossover
                a, b, c = self.select_random_indices(i)
                mutant = self.population[a] + self.F * (self.population[b] - self.population[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                trial = np.copy(self.population[i])
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial[crossover_mask] = mutant[crossover_mask]

                # Quantum-inspired rotation
                phi_rotation = np.random.uniform(0, 2 * np.pi)
                qbit_update = np.cos(phi_rotation) * self.qbits[i] + np.sin(phi_rotation) * (self.population[i] - mutant)
                self.qbits[i] = qbit_update
                quantum_trial = self.lower_bound + (self.upper_bound - self.lower_bound) * (np.cos(qbit_update) + 1) / 2

                # Select the best trial solution
                trial_fitness = func(trial)
                self.evaluations += 1
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness

                quantum_fitness = func(quantum_trial)
                self.evaluations += 1
                if quantum_fitness < self.fitness[i]:
                    self.population[i] = quantum_trial
                    self.fitness[i] = quantum_fitness

                if self.evaluations >= self.budget:
                    break

        return self.best_solution

    def select_random_indices(self, current_index):
        indices = list(range(self.population_size))
        indices.remove(current_index)
        return np.random.choice(indices, 3, replace=False)