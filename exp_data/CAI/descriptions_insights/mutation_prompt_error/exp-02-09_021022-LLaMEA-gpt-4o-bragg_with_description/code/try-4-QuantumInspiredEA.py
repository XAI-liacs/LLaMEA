import numpy as np

class QuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.alpha = 0.5  # Initial quantum state probability amplitude
        self.beta = np.sqrt(1 - self.alpha**2)
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Initialize population in quantum representation
        quantum_population = np.random.rand(self.population_size, self.dim)
        population = self.quantum_measurement(quantum_population, lb, ub)
        fitness = np.array([func(ind) for ind in population])
        self.evaluations += self.population_size
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        while self.evaluations < self.budget:
            # Quantum rotation gate
            self.alpha, self.beta = self.quantum_rotation(self.alpha, self.beta)

            for i in range(self.population_size):
                # Create trial quantum individual
                trial_quantum = self.alpha * quantum_population[i] + self.beta * np.random.randn(self.dim)
                trial_quantum = np.clip(trial_quantum, 0, 1)
                trial = self.quantum_measurement(np.array([trial_quantum]), lb, ub)[0]

                trial_fitness = func(trial)
                self.evaluations += 1
                if trial_fitness < fitness[i]:
                    quantum_population[i] = trial_quantum
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

        return best_solution

    def quantum_measurement(self, quantum_population, lb, ub):
        # Collapse quantum states to real numbers within the search space
        return lb + (ub - lb) * quantum_population

    def quantum_rotation(self, alpha, beta):
        # Adjust quantum state probabilities for exploration-exploitation balance
        theta = np.pi / 100  # Small rotation angle
        new_alpha = alpha * np.cos(theta) - beta * np.sin(theta)
        new_beta = alpha * np.sin(theta) + beta * np.cos(theta)
        return new_alpha, new_beta