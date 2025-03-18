import numpy as np
from scipy.optimize import minimize

class HybridGeneticNelderMead:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        pop_size = max(10, self.dim * 5)
        bounds = np.array([(func.bounds.lb[i], func.bounds.ub[i]) for i in range(self.dim)])
        
        # Initialize population
        population = np.random.uniform(bounds[:, 0], bounds[:, 1], (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.budget -= pop_size
        
        best_idx = np.argmin(fitness)
        best_value = fitness[best_idx]
        best_solution = population[best_idx]

        def select_parents():
            # Tournament selection
            indices = np.random.choice(pop_size, 3, replace=False)
            fit_values = fitness[indices]
            return indices[np.argmin(fit_values)]

        def crossover(parent1, parent2):
            alpha = np.random.rand()
            return alpha * parent1 + (1 - alpha) * parent2

        mutation_rate = 0.1
        
        while self.budget > 0:
            # Create offspring population
            offspring = []
            for _ in range(pop_size):
                if self.budget <= 0:
                    break
                parent1 = population[select_parents()]
                parent2 = population[select_parents()]
                child = crossover(parent1, parent2)
                
                # Mutation
                if np.random.rand() < mutation_rate:
                    mutation_vector = np.random.normal(0, 0.1, self.dim)
                    child = np.clip(child + mutation_vector, bounds[:, 0], bounds[:, 1])
                
                offspring.append(child)
                trial_fitness = func(child)
                self.budget -= 1
                
                if trial_fitness < best_value:
                    best_value = trial_fitness
                    best_solution = child
            
            # Replace population with offspring
            population = np.array(offspring)
            fitness = np.array([func(ind) for ind in population])
            self.budget -= pop_size

            # Local search using Nelder-Mead on the best solution found so far
            if self.budget > 0:
                result = minimize(func, best_solution, method='Nelder-Mead', options={'maxfev': self.budget})
                if result.fun < best_value:
                    best_value = result.fun
                    best_solution = result.x
                    self.budget -= result.nfev
        
        return best_solution