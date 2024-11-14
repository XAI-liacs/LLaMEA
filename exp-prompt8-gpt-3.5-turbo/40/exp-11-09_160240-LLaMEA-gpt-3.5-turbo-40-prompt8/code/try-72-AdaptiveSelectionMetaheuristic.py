import numpy as np

class AdaptiveSelectionMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        pop_size = 10
        scaling_factors = np.full(pop_size, 0.5)
        mutation_rates = np.full(pop_size, 0.5)
        
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        
        for _ in range(self.budget // pop_size):
            population = [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(pop_size)]
            fitness_values = [func(ind) for ind in population]
            
            best_idx = np.argmin(fitness_values)
            population[best_idx] = best_solution
            fitness_values[best_idx] = best_fitness
            
            fitness_values_normalized = 1 / np.array(fitness_values)
            selection_probabilities = fitness_values_normalized / np.sum(fitness_values_normalized)
            
            selected_idx = np.random.choice(range(pop_size), size=pop_size, replace=True, p=selection_probabilities)
            
            for idx, ind in enumerate(population):
                if idx in selected_idx:
                    mutated_solution = ind + scaling_factors[idx] * np.random.normal(0, 1, self.dim)
                    
                    fitness = func(mutated_solution)
                    if fitness < fitness_values[idx]:
                        population[idx] = mutated_solution
                        fitness_values[idx] = fitness
                        scaling_factors[idx] *= 1.1
                        if np.random.uniform(0, 1) < 0.2:
                            mutation_rates[idx] *= 1.2
                        else:
                            mutation_rates[idx] *= 0.9
                    else:
                        scaling_factors[idx] *= 0.9
                        mutation_rates[idx] *= 0.8
                
                    if fitness < best_fitness:
                        best_solution = mutated_solution
                        best_fitness = fitness
                else:
                    scaling_factors[idx] *= 0.9
                    mutation_rates[idx] *= 0.8
        
        return best_solution