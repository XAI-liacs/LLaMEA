import numpy as np

class QuantumGeneticSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)
        self.positions = np.random.rand(self.population_size, dim)
        self.velocities = np.zeros((self.population_size, dim))
        self.pbest = self.positions.copy()
        self.gbest = None
        self.pbest_scores = np.full(self.population_size, float('inf'))
        self.gbest_score = float('inf')
        self.evaluations = 0
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8

    def quantum_mutation(self, position):
        mutation_vector = np.random.uniform(-1, 1, self.dim)
        new_position = position + self.mutation_rate * mutation_vector
        return new_position

    def quantum_crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            alpha = np.random.rand(self.dim)
            child = alpha * parent1 + (1 - alpha) * parent2
            return child
        return parent1

    def _update_particle(self, idx, func):
        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
        self.velocities[idx] = (0.5 * self.velocities[idx] + 
                                r1 * (self.pbest[idx] - self.positions[idx]) +
                                r2 * (self.gbest - self.positions[idx]))
        new_pos = self.positions[idx] + self.velocities[idx]
        new_pos = self.quantum_mutation(new_pos)
        new_pos = np.clip(new_pos, func.bounds.lb, func.bounds.ub)
        new_score = func(new_pos)
        
        if new_score < self.pbest_scores[idx]:
            self.pbest[idx] = new_pos
            self.pbest_scores[idx] = new_score
        
        if new_score < self.gbest_score:
            self.gbest = new_pos
            self.gbest_score = new_score
        
        self.positions[idx] = new_pos
        self.evaluations += 1

    def _perform_crossover(self):
        for i in range(0, self.population_size, 2):
            if i + 1 < self.population_size:
                child1 = self.quantum_crossover(self.positions[i], self.positions[i + 1])
                child2 = self.quantum_crossover(self.positions[i + 1], self.positions[i])
                self.positions[i], self.positions[i + 1] = child1, child2

    def __call__(self, func):
        self.positions = func.bounds.lb + (func.bounds.ub - func.bounds.lb) * np.random.rand(self.population_size, self.dim)
        for i in range(self.population_size):
            score = func(self.positions[i])
            self.pbest[i] = self.positions[i]
            self.pbest_scores[i] = score
            if score < self.gbest_score:
                self.gbest = self.positions[i]
                self.gbest_score = score
            self.evaluations += 1
            if self.evaluations >= self.budget:
                return self.gbest

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                self._update_particle(i, func)
                if self.evaluations >= self.budget:
                    break
            self._perform_crossover()
        
        return self.gbest