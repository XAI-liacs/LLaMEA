import numpy as np

class AdaptiveQuantumPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.swarm_size = 60
        self.initial_inertia = 0.9
        self.final_inertia = 0.3
        self.cognitive_coeff = 1.5
        self.social_coeff = 2.0
        self.quantum_coeff = 0.5
        self.max_velocity = (self.upper_bound - self.lower_bound) * 0.1

    def __call__(self, func):
        np.random.seed(42)
        position = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        velocity = np.random.uniform(-self.max_velocity, self.max_velocity, (self.swarm_size, self.dim))
        personal_best_position = np.copy(position)
        personal_best_value = np.full(self.swarm_size, np.inf)
        global_best_position = None
        global_best_value = np.inf

        evaluations = 0

        while evaluations < self.budget:
            inertia = self.initial_inertia - (self.initial_inertia - self.final_inertia) * (evaluations / self.budget)

            for i in range(self.swarm_size):
                current_value = func(position[i])
                evaluations += 1

                if current_value < personal_best_value[i]:
                    personal_best_value[i] = current_value
                    personal_best_position[i] = position[i].copy()

                if current_value < global_best_value:
                    global_best_value = current_value
                    global_best_position = position[i].copy()

                if evaluations >= self.budget:
                    break

            adaptive_learning_rate = np.clip(1.0 - evaluations / (self.budget * 0.8), 0.3, 1.0)

            for i in range(self.swarm_size):
                r1, r2, r3 = np.random.rand(self.dim), np.random.rand(self.dim), np.random.rand(self.dim)
                inertia_term = inertia * velocity[i]
                cognitive_term = self.cognitive_coeff * r1 * (personal_best_position[i] - position[i])
                social_term = self.social_coeff * r2 * (global_best_position - position[i])

                # Quantum inspired update
                quantum_term = self.quantum_coeff * r3 * (np.random.uniform(self.lower_bound, self.upper_bound, self.dim) - position[i])

                velocity[i] = adaptive_learning_rate * (inertia_term + cognitive_term + social_term + quantum_term)
                velocity[i] = np.clip(velocity[i], -self.max_velocity, self.max_velocity)
                position[i] += velocity[i]
                position[i] = np.clip(position[i], self.lower_bound, self.upper_bound)

        return global_best_value