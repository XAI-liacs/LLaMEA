import numpy as np

class AdaptiveQuantumButterflyOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, dim)
        self.a = 0.5  # Movement intensity constant
        self.c = 0.1  # Sensory modality constant
        self.beta = 0.05  # Quantum-inspired update probability

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_index = np.argmin(fitness)
        best_position = population[best_index].copy()
        evaluations = self.population_size
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(), np.random.rand()
                if r1 < 0.5:
                    # Local search
                    step_size = self.a * r2 * (best_position - population[i])
                else:
                    # Global search
                    random_partner = population[np.random.randint(self.population_size)]
                    step_size = self.c * r2 * (random_partner - population[i])
                
                # Quantum-inspired perturbation
                if np.random.rand() < self.beta:
                    q = np.random.normal(loc=0, scale=1)
                    step_size += q * (ub - lb)
                
                # Update position
                population[i] = np.clip(population[i] + step_size, lb, ub)
                
                # Evaluate new position
                new_fitness = func(population[i])
                evaluations += 1

                # Update the best solution
                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness
                    if new_fitness < fitness[best_index]:
                        best_index = i
                        best_position = population[i].copy()

                if evaluations >= self.budget:
                    break
                    
        return best_position, fitness[best_index]