import numpy as np

class QuantumInspiredEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_individuals = 20
        self.q_population = None
        self.best_position = None
        self.best_score = float('inf')
        self.alpha = 0.02  # Rotation angle parameter
    
    def initialize_population(self):
        self.q_population = np.random.rand(self.num_individuals, self.dim) * 2 * np.pi  # Initialize quantum angles
    
    def measure(self):
        # Convert quantum bits to binary solutions
        return np.random.rand(self.num_individuals, self.dim) < 0.5 * (1 + np.sin(self.q_population))
    
    def update_quantum_population(self, measured_population, scores):
        for i in range(self.num_individuals):
            for d in range(self.dim):
                prob_0 = 0.5 * (1 + np.sin(self.q_population[i, d]))
                if measured_population[i, d] == 0:
                    rotation_direction = 1 if np.random.rand() > prob_0 else -1
                else:
                    rotation_direction = -1 if np.random.rand() > prob_0 else 1
                # Update quantum population using a rotation gate
                self.q_population[i, d] += rotation_direction * self.alpha
    
    def __call__(self, func):
        self.initialize_population()
        evaluations = 0
        
        while evaluations < self.budget:
            measured_population = self.measure()
            scores = np.array([func(measured_population[i]) for i in range(self.num_individuals)])
            evaluations += self.num_individuals
            
            # Update best solution found
            for i in range(self.num_individuals):
                if scores[i] < self.best_score:
                    self.best_score = scores[i]
                    self.best_position = measured_population[i]
            
            self.update_quantum_population(measured_population, scores)