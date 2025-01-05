import numpy as np

class QuantumInspiredDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.history = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(x) for x in pop])
        best_solution = pop[np.argmin(fitness)]
        
        evaluations = self.population_size

        while evaluations < self.budget:
            new_pop = np.zeros_like(pop)
            for i in range(self.population_size):
                idxs = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = pop[idxs]
                mutant = self.quantum_superposition(a, b, c, lb, ub)
                
                crossover = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(crossover, mutant, pop[i])
                
                trial = np.clip(trial, lb, ub)
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    new_pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < func(best_solution):
                        best_solution = trial
                else:
                    new_pop[i] = pop[i]

            pop = new_pop
            self.history.append(best_solution)
            if evaluations >= self.budget:
                break

        return best_solution

    def quantum_superposition(self, a, b, c, lb, ub):
        # Generate random coefficients for quantum-inspired linear superposition
        alpha, beta, gamma = np.random.rand(3)
        total = alpha + beta + gamma
        alpha, beta, gamma = alpha/total, beta/total, gamma/total
        
        # Create superposition of vectors a, b, c
        superposed = alpha * a + beta * b + gamma * c
        superposed += self.mutation_factor * (b - c)
        
        # Clip to bounds and return
        return np.clip(superposed, lb, ub)