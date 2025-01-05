import numpy as np

class Predator_Prey_Swarm_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 30
        self.predator_count = 5
        self.c1 = 1.5
        self.c2 = 1.5
        self.alpha = 0.5
        self.beta = 0.3
        self.predator_effect = 1.0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)

        position = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_position = np.copy(position)
        personal_best_value = np.array([func(p) for p in personal_best_position])
        global_best_position = personal_best_position[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)

        evaluations = self.swarm_size

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                if i < self.predator_count:
                    # Predator dynamics
                    velocity[i] += self.predator_effect * np.random.normal(size=self.dim)
                else:
                    # Prey dynamics
                    r1, r2 = np.random.rand(2)
                    velocity[i] = (self.alpha * velocity[i] +
                                   self.c1 * r1 * (personal_best_position[i] - position[i]) +
                                   self.c2 * r2 * (global_best_position - position[i]))

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

            # Update predator effect dynamically
            self.predator_effect *= (1.0 - self.beta)

        return global_best_position, global_best_value