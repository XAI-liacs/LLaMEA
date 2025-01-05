import numpy as np

class Adaptive_Swarm_Quantum_Collaborative_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.c1 = 2.0
        self.c2 = 2.0
        self.inertia_weight = 0.9
        self.collab_weight = 0.5
        self.inertia_damping = 0.99

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)

        position = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_position = np.copy(position)
        personal_best_value = np.array([func(p) for p in personal_best_position])
        global_best_position = personal_best_position[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                inertia = self.inertia_weight * velocity[i]
                cognitive = self.c1 * r1 * (personal_best_position[i] - position[i])
                social = self.c2 * r2 * (global_best_position - position[i])
                
                # Quantum-inspired collaboration
                partner_index = np.random.randint(self.population_size)
                collaborative = self.collab_weight * (position[partner_index] - position[i])

                velocity[i] = inertia + cognitive + social + collaborative
                position[i] += velocity[i]
                position[i] = np.clip(position[i], lb, ub)

                current_value = func(position[i])
                evaluations += 1

                if current_value < personal_best_value[i]:
                    personal_best_position[i] = position[i]
                    personal_best_value[i] = current_value
                
                if current_value < global_best_value:
                    global_best_position = position[i]
                    global_best_value = current_value

                if evaluations >= self.budget:
                    break

            self.inertia_weight *= self.inertia_damping

        return global_best_position, global_best_value