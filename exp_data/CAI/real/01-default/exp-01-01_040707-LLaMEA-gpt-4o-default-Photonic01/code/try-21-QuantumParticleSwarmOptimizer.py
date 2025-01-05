import numpy as np

class QuantumParticleSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.7
        self.position = None
        self.velocity = None
        self.pbest = None
        self.pbest_scores = None
        self.gbest = None
        self.gbest_score = float('inf')
        self.alpha = 0.5  # Quantum adaptation parameter
        self.beta = 0.5   # Quantum crossover probability

    def initialize(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        self.position = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.velocity = np.random.rand(self.population_size, self.dim) - 0.5
        self.pbest = np.copy(self.position)
        self.pbest_scores = np.full(self.population_size, float('inf'))

    def evaluate(self, func):
        scores = np.array([func(p) for p in self.position])
        for i in range(self.population_size):
            if scores[i] < self.pbest_scores[i]:
                self.pbest_scores[i] = scores[i]
                self.pbest[i] = self.position[i]
            if scores[i] < self.gbest_score:
                self.gbest_score = scores[i]
                self.gbest = self.position[i]
        return scores

    def update_velocity_position(self):
        r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
        cognitive = self.c1 * r1 * (self.pbest - self.position)
        social = self.c2 * r2 * (self.gbest - self.position)
        self.velocity = self.w * self.velocity + cognitive + social
        self.position += self.velocity

    def quantum_superposition(self, bounds):
        lb, ub = np.array(bounds.lb), np.array(bounds.ub)
        new_population = np.copy(self.position)
        for i in range(self.population_size):
            qbit = np.random.rand(self.dim) < self.beta
            q_position = lb + (ub - lb) * np.random.rand(self.dim)
            new_position = np.where(qbit, q_position, self.position[i])
            new_population[i] = new_position
        return new_population

    def __call__(self, func):
        func_calls = 0
        self.initialize(func.bounds)
        max_iterations = self.budget // self.population_size
        iteration = 0
        
        while func_calls < self.budget:
            scores = self.evaluate(func)
            func_calls += self.population_size
            
            self.update_velocity_position()
            quantum_population = self.quantum_superposition(func.bounds)
            scores_quantum = np.array([func(p) for p in quantum_population])
            func_calls += self.population_size
            
            self.position = np.where(
                scores[:, np.newaxis] <= scores_quantum[:, np.newaxis],
                self.position, quantum_population
            )
            
            # Adaptive parameter control
            if iteration % 10 == 0:
                self.w = max(0.4, self.w * 0.95)
                self.alpha = min(0.9, self.alpha + 0.05)
                self.beta = min(0.9, self.beta + 0.02)
                
            iteration += 1
        
        return self.gbest, self.gbest_score