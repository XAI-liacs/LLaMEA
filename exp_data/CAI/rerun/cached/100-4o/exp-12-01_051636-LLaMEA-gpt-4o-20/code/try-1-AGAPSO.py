import numpy as np

class AGAPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.num_particles = 50
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.learning_rate = 0.01
        self.grad_epsilon = 1e-8

    def _initialize_particles(self):
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        return positions, velocities

    def _calculate_gradients(self, func, positions, current_best):
        gradients = []
        for i, position in enumerate(positions):
            grad = np.zeros(self.dim)
            for j in range(self.dim):
                perturbed_position = np.copy(position)
                perturbed_position[j] += self.grad_epsilon
                grad[j] = (func(perturbed_position) - func(position)) / self.grad_epsilon
            gradients.append(grad)
        gradients = np.array(gradients)
        return (current_best - positions) * gradients
    
    def __call__(self, func):
        positions, velocities = self._initialize_particles()
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.num_particles, float('inf'))
        global_best_position = None
        global_best_score = float('inf')

        evaluations = 0

        while evaluations < self.budget:
            scores = np.array([func(pos) for pos in positions])
            evaluations += self.num_particles

            better_personal_mask = scores < personal_best_scores
            personal_best_scores[better_personal_mask] = scores[better_personal_mask]
            personal_best_positions[better_personal_mask] = positions[better_personal_mask]

            if np.min(scores) < global_best_score:
                global_best_position = positions[np.argmin(scores)]
                global_best_score = np.min(scores)

            gradients = self._calculate_gradients(func, positions, global_best_position)
            velocities = (
                self.inertia_weight * velocities +
                self.cognitive_coeff * np.random.rand(self.num_particles, self.dim) * (personal_best_positions - positions) +
                self.social_coeff * np.random.rand(self.num_particles, self.dim) * (global_best_position - positions) +
                self.learning_rate * gradients
            )
            positions = positions + velocities
            positions = np.clip(positions, self.lower_bound, self.upper_bound)

        return global_best_position, global_best_score