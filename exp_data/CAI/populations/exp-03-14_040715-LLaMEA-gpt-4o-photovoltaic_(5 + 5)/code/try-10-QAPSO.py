import numpy as np

class QAPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30  # Customizable
        self.inertia_weight = 0.9
        self.cognition_learning_factor = 1.5
        self.social_learning_factor = 2.0
        self.quantum_factor = 0.5  # Controls the spread of the quantum superposition

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        position = np.random.uniform(lb, ub, (self.population_size, self.dim))
        personal_best_position = position.copy()
        personal_best_score = np.array([func(p) for p in position])
        global_best_position = personal_best_position[personal_best_score.argmin()]
        global_best_score = personal_best_score.min()

        evaluations = self.population_size

        while evaluations < self.budget:
            r_p = np.random.uniform(0, 1, (self.population_size, self.dim))
            r_g = np.random.uniform(0, 1, (self.population_size, self.dim))

            # Update inertia weight dynamically
            self.inertia_weight = 0.4 + 0.5 * ((self.budget - evaluations) / self.budget)
            
            # Update velocities with quantum-inspired dynamics
            velocity = (self.inertia_weight * velocity
                        + self.cognition_learning_factor * r_p * (personal_best_position - position)
                        + self.social_learning_factor * r_g * (global_best_position - position)
                        + self.quantum_factor * np.random.uniform(-1, 1, (self.population_size, self.dim)))

            # Update positions
            position += velocity
            position = np.clip(position, lb, ub)
            
            # Opposition-based learning
            opposition_position = lb + ub - position
            opposition_position = np.clip(opposition_position, lb, ub)
            opposition_scores = np.array([func(p) for p in opposition_position])
            
            # Evaluate new solutions
            scores = np.array([func(p) for p in position])
            evaluations += self.population_size

            # Combine scores and positions
            combined_scores = np.concatenate((scores, opposition_scores))
            combined_positions = np.vstack((position, opposition_position))

            # Update personal bests
            improved = combined_scores < personal_best_score
            personal_best_position[improved[:self.population_size]] = combined_positions[improved][:self.population_size]
            personal_best_score[improved[:self.population_size]] = combined_scores[improved][:self.population_size]

            # Update global best
            if combined_scores.min() < global_best_score:
                global_best_score = combined_scores.min()
                global_best_position = combined_positions[combined_scores.argmin()]

        return global_best_position, global_best_score