import numpy as np

class QuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.q_population = np.random.rand(self.population_size, self.dim) * 2 * np.pi
        self.binary_population = np.sign(np.sin(self.q_population))
        self.best_solution = None
        self.best_score = np.inf
        self.mutation_rate = 0.1
        self.adapt_rate = 0.05
    
    def _quantum_to_real(self, q):
        return 0.5 * (1 + np.sign(np.sin(q)))
    
    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        eval_count = 0
        
        while eval_count < self.budget:
            real_population = lb + (ub - lb) * self._quantum_to_real(self.q_population)
            scores = np.apply_along_axis(func, 1, np.clip(real_population, lb, ub))
            eval_count += self.population_size
            
            best_idx = np.argmin(scores)
            if scores[best_idx] < self.best_score:
                self.best_score = scores[best_idx]
                self.best_solution = real_population[best_idx]
            
            for i in range(self.population_size):
                diff = self.best_solution - real_population[i]
                self.q_population[i] += self.adapt_rate * diff * np.random.rand(self.dim)
                self.q_population[i] = np.mod(self.q_population[i], 2 * np.pi)
                
                if np.random.rand() < self.mutation_rate:
                    mutation = np.random.rand(self.dim) * 2 * np.pi
                    self.q_population[i] = np.mod(self.q_population[i] + mutation, 2 * np.pi)
        
        return self.best_solution, self.best_score