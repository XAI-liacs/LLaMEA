import numpy as np

class HQASO:
    def __init__(self, budget, dim, harmony_memory_size=10, swarm_size=30, quantum_prob=0.2, adaptation_rate=0.1):
        self.budget = budget
        self.dim = dim
        self.harmony_memory_size = harmony_memory_size
        self.swarm_size = swarm_size
        self.quantum_prob = quantum_prob
        self.adaptation_rate = adaptation_rate
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        harmony_memory = [self.initialize_solution(lb, ub) for _ in range(self.harmony_memory_size)]
        harmony_values = [func(harmony) for harmony in harmony_memory]
        global_best_index = np.argmin(harmony_values)
        best_global_position = harmony_memory[global_best_index].copy()
        best_global_value = harmony_values[global_best_index]

        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        positions = self.initialize_swarm(lb, ub)

        while self.evaluations < self.budget:
            for i in range(self.swarm_size):
                if np.random.rand() < self.quantum_prob:
                    positions[i] = self.quantum_perturbation(positions[i], lb, ub)
                
                if np.random.rand() < 0.5:
                    harmony_choice = harmony_memory[np.random.choice(self.harmony_memory_size)]
                    new_position = np.clip(positions[i] + self.adaptation_rate * (harmony_choice - positions[i]), lb, ub)
                else:
                    new_position = np.clip(positions[i] + velocities[i], lb, ub)

                value = func(new_position)
                self.evaluations += 1

                if value < best_global_value:
                    best_global_value = value
                    best_global_position = new_position.copy()

                if value < harmony_values[global_best_index]:
                    global_best_index = np.argmin(harmony_values)
                    harmony_memory[global_best_index] = new_position
                    harmony_values[global_best_index] = value

                velocities[i] = self.adaptive_velocity_update(positions[i], best_global_position, harmony_memory, velocities[i], lb, ub)
                positions[i] = new_position

                if self.evaluations >= self.budget:
                    break

        return best_global_position

    def initialize_solution(self, lb, ub):
        return np.random.uniform(lb, ub, self.dim)

    def initialize_swarm(self, lb, ub):
        return np.random.uniform(lb, ub, (self.swarm_size, self.dim))

    def quantum_perturbation(self, position, lb, ub):
        q_position = position + (np.random.rand(self.dim) - 0.5) * (ub - lb) * 0.1
        return np.clip(q_position, lb, ub)

    def adaptive_velocity_update(self, position, best_global_position, harmony_memory, velocity, lb, ub):
        inertia = 0.5
        cognitive = 1.5
        social = 1.5
        harmony_influence = np.random.rand() * (harmony_memory[np.random.choice(self.harmony_memory_size)] - position)
        cognitive_component = cognitive * np.random.random(self.dim) * (position - best_global_position)
        social_component = social * np.random.random(self.dim) * harmony_influence
        new_velocity = inertia * velocity + cognitive_component + social_component
        return np.clip(new_velocity, -1, 1)