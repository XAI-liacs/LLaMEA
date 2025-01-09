import numpy as np

class QuantumInspiredEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, budget // 10)
        self.quantum_bits = np.random.rand(self.population_size, dim)
        self.population = np.zeros((self.population_size, dim))
        self.best_solution = np.zeros(dim)
        self.best_score = float('inf')
        self.alpha = 0.75  # Quantum interference control
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        
        def measure(qubit):
            return np.where(np.random.rand(self.dim) < qubit, 1, 0)
        
        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.population_size):
                self.population[i] = measure(self.quantum_bits[i])
                self.population[i] = lb + self.population[i] * (ub - lb)
                score = func(self.population[i])
                evaluations += 1
                if score < self.best_score:
                    self.best_score = score
                    self.best_solution = np.copy(self.population[i])
            
            # Quantum rotation gate update
            for i in range(self.population_size):
                probability_update = self.alpha * (self.best_solution - self.population[i]) / (ub - lb)
                self.quantum_bits[i] = np.clip(self.quantum_bits[i] + probability_update, 0, 1)
                
        return self.best_solution