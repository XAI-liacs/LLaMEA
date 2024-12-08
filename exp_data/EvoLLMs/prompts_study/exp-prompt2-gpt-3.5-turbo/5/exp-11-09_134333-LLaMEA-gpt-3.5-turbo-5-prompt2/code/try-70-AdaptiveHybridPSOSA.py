import numpy as np

class AdaptiveHybridPSOSA(HybridPSOSA):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
    
    def __call__(self, func):
        def adaptive_mutation(best_solution, inertia_weight, best_fitness):
            mutation_rate = np.random.uniform(0.1, 1.0)
            if np.random.rand() < 0.5 or func(best_solution + np.random.normal(0, 1.0, self.dim)) < best_fitness:
                new_solution = best_solution + mutation_rate * np.random.uniform(-1.0, 1.0, self.dim) * inertia_weight
            else:
                new_solution = best_solution + mutation_rate * np.random.normal(0, 1.0, self.dim)
            return new_solution
        
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        inertia_weight = 0.5  # Initial inertia weight
        
        for _ in range(self.budget):
            new_solution = adaptive_mutation(best_solution, inertia_weight, best_fitness)
            new_fitness = func(new_solution)
            
            if new_fitness < best_fitness:
                best_solution = new_solution
                best_fitness = new_fitness
                inertia_weight = max(0.4, inertia_weight * 0.99)  # Update inertia weight dynamically
        
        return best_solution