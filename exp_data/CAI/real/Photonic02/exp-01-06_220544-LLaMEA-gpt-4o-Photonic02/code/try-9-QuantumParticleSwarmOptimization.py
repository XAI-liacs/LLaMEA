import numpy as np

class QuantumParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 50
        self.q_population = np.random.rand(self.swarm_size, self.dim)
        self.velocities = np.random.rand(self.swarm_size, self.dim) * 0.1
        self.mutation_rate = 0.05
        self.global_best_position = np.zeros(self.dim)
        self.global_best_score = float('inf')

    def qbit_to_solution(self):
        return np.random.rand(self.swarm_size, self.dim) < self.q_population

    def evaluate_population(self, func, population):
        return np.array([func(indiv) for indiv in population])

    def update_q_population(self):
        for i in range(self.swarm_size):
            if self.local_best_scores[i] < self.global_best_score:
                self.global_best_score = self.local_best_scores[i]
                self.global_best_position = self.local_best_positions[i]

    def __call__(self, func):
        current_eval = 0

        local_best_positions = np.copy(self.q_population)
        local_best_scores = np.full(self.swarm_size, float('inf'))
        
        while current_eval < self.budget:
            binary_population = self.qbit_to_solution()
            population = binary_population * (func.bounds.ub - func.bounds.lb) + func.bounds.lb
            scores = self.evaluate_population(func, population)
            current_eval += len(scores)
            
            for i in range(self.swarm_size):
                if scores[i] < local_best_scores[i]:
                    local_best_scores[i] = scores[i]
                    local_best_positions[i] = binary_population[i]
            
            self.update_q_population()

            for i in range(self.swarm_size):
                cognitive = np.random.rand(self.dim) * (local_best_positions[i] - self.q_population[i])
                social = np.random.rand(self.dim) * (self.global_best_position - self.q_population[i])
                self.velocities[i] = self.velocities[i] + cognitive + social
                self.q_population[i] += self.velocities[i]
                mutation_mask = np.random.rand(self.dim) < self.mutation_rate
                self.q_population[i][mutation_mask] = np.random.rand(np.sum(mutation_mask))
                
        return self.global_best_position