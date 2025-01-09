import numpy as np

class HybridFireflySA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def __call__(self, func):
        def firefly_search(pop_size, alpha=0.1, beta0=1.0, gamma=0.1):
            def move_firefly(firefly, best_firefly):
                beta = beta0 * np.exp(-gamma * np.linalg.norm(firefly - best_firefly))
                return firefly + alpha * (np.random.rand(self.dim) - 0.5) + beta * (best_firefly - firefly) + 0.01 * np.random.randn(self.dim)
            
            pop = np.random.uniform(-5.0, 5.0, size=(pop_size, self.dim))
            best_firefly = min(pop, key=func)
            
            for _ in range(self.budget):
                new_pop = [move_firefly(firefly, best_firefly) for firefly in pop]
                pop = np.clip(new_pop, -5.0, 5.0)
                best_firefly = min(pop, key=func)
            
            return best_firefly
        
        def simulated_annealing(init_solution, T=1.0, alpha=0.9):
            current_solution = init_solution
            best_solution = current_solution
            
            for _ in range(int(self.budget)):
                new_solution = current_solution + 0.1 * np.random.randn(self.dim)
                delta_E = func(new_solution) - func(current_solution)
                
                if delta_E < 0 or np.random.rand() < np.exp(-delta_E / T):
                    current_solution = new_solution
                
                if func(current_solution) < func(best_solution):
                    best_solution = current_solution
                
                T *= alpha
            
            return best_solution
        
        return firefly_search(20)  # Hybrid optimization using Firefly Algorithm with Simulated Annealing