import numpy as np

class QuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.alpha = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.history = []  # Keep track of best solutions

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(x) for x in pop])
        best_global = pop[np.argmin(fitness)]
        evaluations = self.population_size

        while evaluations < self.budget:
            new_pop = np.copy(pop)
            for i in range(self.population_size):
                # Quantum-inspired superposition
                r1, r2, r3 = np.random.choice(self.population_size, 3, replace=False)
                mutant_vector = pop[r1] + self.alpha * (pop[r2] - pop[r3])
                mutant_vector = np.clip(mutant_vector, lb, ub)
                
                # Entanglement and crossover
                trial_vector = np.copy(pop[i])
                crossover_indices = np.random.rand(self.dim) < self.CR
                trial_vector[crossover_indices] = mutant_vector[crossover_indices]

                # Selection
                trial_fitness = func(trial_vector)
                if trial_fitness < fitness[i]:
                    new_pop[i] = trial_vector
                    fitness[i] = trial_fitness

                # Update global best
                if trial_fitness < func(best_global):
                    best_global = trial_vector

            pop = new_pop
            evaluations += self.population_size
            self.history.append(best_global)

        return best_global