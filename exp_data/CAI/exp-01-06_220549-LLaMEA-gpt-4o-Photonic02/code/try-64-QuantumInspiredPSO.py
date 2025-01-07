import numpy as np

class QuantumInspiredPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(30, self.budget // (2 * dim))  # Dynamic population size
        self.phi = 0.5 + np.log(self.dim) / np.log(2)  # Constriction factor for convergence
        self.gamma = 0.5  # Balance between exploration and exploitation
        self.positions = np.random.rand(self.population_size, self.dim)  # Initialize particle positions
        self.velocities = np.random.rand(self.population_size, self.dim)  # Initialize particle velocities
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, np.inf)

    def __call__(self, func):
        global_best_position = None
        global_best_score = np.inf
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Boundary handling
                self.positions[i] = np.clip(self.positions[i], func.bounds.lb, func.bounds.ub)

                # Evaluate the fitness
                score = func(self.positions[i])
                evaluations += 1

                # Update personal best
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i]

                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = self.positions[i]

                if evaluations >= self.budget:
                    break

            # Update velocities and positions
            if evaluations < self.budget:
                for i in range(self.population_size):
                    # Quantum-inspired position update
                    self.velocities[i] = self.phi * (self.velocities[i] 
                                  + ((1 - evaluations/self.budget) * self.gamma) * (self.personal_best_positions[i] - self.positions[i]) 
                                  + self.gamma * (global_best_position - self.positions[i]))
                    
                    self.positions[i] += self.velocities[i]

        return global_best_position, global_best_score