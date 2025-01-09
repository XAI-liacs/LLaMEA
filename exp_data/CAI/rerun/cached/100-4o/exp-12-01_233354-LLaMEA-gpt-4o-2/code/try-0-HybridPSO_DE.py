import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 20
        self.inertia_weight = 0.729  # Commonly used inertia weight
        self.cognitive_coeff = 1.4944  # Cognitive coefficient
        self.social_coeff = 1.4944    # Social coefficient
        self.bounds = (-5.0, 5.0)
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9

    def __call__(self, func):
        np.random.seed(42)  # For reproducibility
        position = np.random.uniform(self.bounds[0], self.bounds[1], (self.swarm_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_position = np.copy(position)
        personal_best_value = np.array([func(p) for p in personal_best_position])
        global_best_index = np.argmin(personal_best_value)
        global_best_position = personal_best_position[global_best_index]
        global_best_value = personal_best_value[global_best_index]
        
        evaluations = self.swarm_size

        while evaluations < self.budget:
            # Update velocities and positions
            for i in range(self.swarm_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                velocity[i] = (self.inertia_weight * velocity[i] +
                               self.cognitive_coeff * r1 * (personal_best_position[i] - position[i]) +
                               self.social_coeff * r2 * (global_best_position - position[i]))
                position[i] = position[i] + velocity[i]
                position[i] = np.clip(position[i], self.bounds[0], self.bounds[1])

            # Evaluate and update personal best
            for i in range(self.swarm_size):
                if evaluations >= self.budget:
                    break
                current_value = func(position[i])
                evaluations += 1
                if current_value < personal_best_value[i]:
                    personal_best_value[i] = current_value
                    personal_best_position[i] = position[i]

            # Update global best
            current_global_best_index = np.argmin(personal_best_value)
            if personal_best_value[current_global_best_index] < global_best_value:
                global_best_value = personal_best_value[current_global_best_index]
                global_best_position = personal_best_position[current_global_best_index]

            # Differential Evolution mutation and crossover
            for i in range(self.swarm_size):
                if evaluations >= self.budget:
                    break
                indices = [idx for idx in range(self.swarm_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant_vector = personal_best_position[a] + self.mutation_factor * (personal_best_position[b] - personal_best_position[c])
                mutant_vector = np.clip(mutant_vector, self.bounds[0], self.bounds[1])
                
                trial_vector = np.copy(position[i])
                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                trial_vector[crossover_mask] = mutant_vector[crossover_mask]

                trial_value = func(trial_vector)
                evaluations += 1
                if trial_value < personal_best_value[i]:
                    personal_best_value[i] = trial_value
                    personal_best_position[i] = trial_vector

                    if trial_value < global_best_value:
                        global_best_value = trial_value
                        global_best_position = trial_vector

        return global_best_value