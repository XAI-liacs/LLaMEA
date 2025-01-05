import numpy as np

class QuantumInspiredDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50  # Size of population
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.quantum_rotation_angle = np.pi / 4  # Quantum rotation angle

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(x) for x in pop])
        best_global = pop[np.argmin(fitness)]

        evaluations = self.population_size

        while evaluations < self.budget:
            new_pop = np.copy(pop)
            for i in range(self.population_size):
                # Mutation operation
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = pop[a] + self.F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)

                # Crossover operation
                trial = np.copy(pop[i])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == j_rand:
                        trial[j] = mutant[j]

                # Evaluate the trial solution
                f_trial = func(trial)
                if f_trial < fitness[i]:
                    new_pop[i] = trial
                    fitness[i] = f_trial

                # Quantum-inspired update using superposition principle
                phi = self.quantum_rotation_angle
                for d in range(self.dim):
                    theta = np.random.rand() * 2 * np.pi
                    if np.random.rand() < 0.5:
                        new_pop[i, d] = pop[i, d] * np.cos(phi) + best_global[d] * np.sin(phi)
                    else:
                        new_pop[i, d] = pop[i, d] * np.cos(phi) - best_global[d] * np.sin(phi)

                new_pop[i] = np.clip(new_pop[i], lb, ub)
                fitness[i] = func(new_pop[i])
                
                # Update the global best
                if fitness[i] < func(best_global):
                    best_global = new_pop[i]

            pop = new_pop
            evaluations += self.population_size

        return best_global