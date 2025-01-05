import numpy as np

class AdaptiveQuantumDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.alpha = 0.5  # Quantum rotation factor

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        quantum_population = np.random.rand(self.population_size, self.dim)
        position_population = self.quantum_to_position(quantum_population, lb, ub)
        fitness = np.array([func(ind) for ind in position_population])
        evaluations = self.population_size
        best_index = np.argmin(fitness)
        best_position = position_population[best_index]

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Mutation: create a mutant vector
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = quantum_population[np.random.choice(idxs, 3, replace=False)]
                mutant = a + self.F * (b - c)
                mutant = np.clip(mutant, 0, 1)

                # Crossover: mix mutant with parent
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial_quantum = np.where(cross_points, mutant, quantum_population[i])

                # Convert to position and evaluate
                trial_position = self.quantum_to_position(trial_quantum, lb, ub)
                new_fitness = func(trial_position)
                evaluations += 1

                # Selection: replace if trial is better
                if new_fitness < fitness[i]:
                    quantum_population[i] = trial_quantum
                    position_population[i] = trial_position
                    fitness[i] = new_fitness

                # Update best solution
                if new_fitness < fitness[best_index]:
                    best_index = i
                    best_position = trial_position

                if evaluations >= self.budget:
                    break

            # Adaptive mechanism
            self.F = np.random.uniform(0.4, 0.9)
            self.CR = np.random.uniform(0.7, 1.0)

        return best_position, fitness[best_index]

    def quantum_to_position(self, quantum_bits, lb, ub):
        position = lb + quantum_bits * (ub - lb)
        return position