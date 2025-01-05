import numpy as np

class Adaptive_Quantum_Differential_Evolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 40
        self.F = 0.5  # Mutation factor
        self.CR = 0.9  # Crossover probability
        self.q_scale = 0.05  # Quantum mutation scale
        self.adaptive_factor = 1.05  # Adaptive increase on stagnation

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)

        # Initialize population
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        
        evaluations = self.population_size

        while evaluations < self.budget:
            new_population = np.empty_like(population)
            for i in range(self.population_size):
                # Mutation using differential evolution strategy
                candidates = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[candidates]
                mutant = a + self.F * (b - c)
                mutant = np.clip(mutant, lb, ub)
                
                # Crossover
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, population[i])
                
                # Apply quantum mutation for exploration
                quantum_mutation = self.q_scale * np.random.normal(0, 1, self.dim)
                trial += quantum_mutation
                trial = np.clip(trial, lb, ub)
                
                # Evaluate trial individual
                trial_fitness = func(trial)
                evaluations += 1

                # Selection
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                else:
                    new_population[i] = population[i]
                    
                # Update the best solution found
                if trial_fitness < best_fitness:
                    best_solution = trial
                    best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

            population = new_population

            # Adapt the mutation factor if no improvement
            if np.min(fitness) >= best_fitness:
                self.F *= self.adaptive_factor

        return best_solution, best_fitness