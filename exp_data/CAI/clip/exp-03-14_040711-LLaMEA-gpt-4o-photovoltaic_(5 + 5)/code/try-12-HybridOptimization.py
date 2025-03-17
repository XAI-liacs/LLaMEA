import numpy as np

class HybridOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize parameters
        population_size = 10 * self.dim
        crossover_rate = 0.9
        mutation_factor = 0.8

        # Generate initial population
        population = np.random.rand(population_size, self.dim)
        population = func.bounds.lb + population * (func.bounds.ub - func.bounds.lb)
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        best_idx = np.argmin(fitness)
        best_ind = population[best_idx]
        best_fit = fitness[best_idx]

        # Elite selection: retain the best solution
        elite_ind = best_ind.copy()
        elite_fit = best_fit
        
        while evaluations < self.budget:
            # Adjust mutation factor dynamically, considering convergence
            mutation_factor = 0.4 + 0.4 * (self.budget - evaluations) / self.budget
            
            for i in range(population_size):
                if evaluations >= self.budget:
                    break

                # Differential Evolution mutation and crossover with dynamic mutation factor
                indices = list(range(population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = np.clip(population[a] + mutation_factor * (population[b] - population[c]), func.bounds.lb, func.bounds.ub)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < crossover_rate
                trial = np.where(cross_points, mutant, population[i])
                
                # Evaluate trial
                trial_fitness = func(trial)
                evaluations += 1

                # Select based on fitness
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    
                    # Update the best individual
                    if trial_fitness < best_fit:
                        best_ind = trial
                        best_fit = trial_fitness

            # Adaptive random search
            for _ in range(population_size // 5):
                if evaluations >= self.budget:
                    break

                candidate = np.random.randn(self.dim) * 0.1 + elite_ind
                candidate = np.clip(candidate, func.bounds.lb, func.bounds.ub)
                candidate_fitness = func(candidate)
                evaluations += 1
                
                # Elite replacement strategy
                if candidate_fitness < elite_fit:
                    elite_ind = candidate
                    elite_fit = candidate_fitness

                if candidate_fitness < best_fit:
                    best_ind = candidate
                    best_fit = candidate_fitness

        return best_ind