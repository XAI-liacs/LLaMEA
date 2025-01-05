import numpy as np

class QuantumEnhancedPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.positions = None
        self.velocities = None
        self.best_personal_positions = None
        self.best_personal_scores = None
        self.best_global_position = None
        self.best_global_score = float('inf')
        self.initial_inertia_weight = 0.9
        self.final_inertia_weight = 0.4
        self.cognitive_coefficient = 1.5
        self.social_coefficient = 1.5

    def initialize_swarm(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.best_personal_positions = np.copy(self.positions)
        self.best_personal_scores = np.full(self.population_size, float('inf'))

    def quantum_perturbation(self, position):
        quantum_bit = np.random.rand(self.dim) < 0.5
        quantum_flip = np.where(quantum_bit, 1, -1)
        perturbed_position = position + 0.1 * quantum_flip
        return perturbed_position

    def __call__(self, func):
        self.initialize_swarm(func.bounds)
        evaluations = 0
        lb, ub = func.bounds.lb, func.bounds.ub

        while evaluations < self.budget:
            inertia_weight = self.initial_inertia_weight - (self.initial_inertia_weight - self.final_inertia_weight) * (evaluations / self.budget)
            
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                score = func(self.positions[i])
                evaluations += 1

                if score < self.best_personal_scores[i]:
                    self.best_personal_scores[i] = score
                    self.best_personal_positions[i] = self.positions[i]

                if score < self.best_global_score:
                    self.best_global_score = score
                    self.best_global_position = self.positions[i]

            for i in range(self.population_size):
                cognitive_velocity = self.cognitive_coefficient * np.random.rand(self.dim) * (self.best_personal_positions[i] - self.positions[i])
                social_velocity = self.social_coefficient * np.random.rand(self.dim) * (self.best_global_position - self.positions[i])
                self.velocities[i] = inertia_weight * self.velocities[i] + cognitive_velocity + social_velocity
                self.positions[i] = np.clip(self.positions[i] + self.velocities[i], lb, ub)

            # Quantum perturbation for diversity
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                perturbed_position = self.quantum_perturbation(self.positions[i])
                perturbed_position = np.clip(perturbed_position, lb, ub)
                perturbed_score = func(perturbed_position)
                evaluations += 1

                if perturbed_score < self.best_global_score:
                    self.best_global_score = perturbed_score
                    self.best_global_position = perturbed_position

        return self.best_global_position, self.best_global_score