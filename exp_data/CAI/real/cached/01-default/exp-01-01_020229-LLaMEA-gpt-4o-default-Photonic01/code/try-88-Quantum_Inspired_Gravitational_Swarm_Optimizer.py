import numpy as np

class Quantum_Inspired_Gravitational_Swarm_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.G0 = 100  # Initial gravitational constant
        self.alpha = 20.0  # Decay factor for gravitational constant
        self.quantum_scale = 0.05
        self.local_perturbation_strength = 0.1
        self.inertia_weight = 0.7

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
            gravitational_constant = self.G0 * np.exp(-self.alpha * evaluations / self.budget)
            
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if personal_best_value[j] < personal_best_value[i]:
                        r_ij = np.linalg.norm(personal_best_position[j] - personal_best_position[i])
                        force = gravitational_constant * (personal_best_position[j] - personal_best_position[i]) / (r_ij + 1e-12)
                        velocity[i] += force + np.random.normal(scale=self.quantum_scale, size=self.dim)
                
                r1 = np.random.rand()
                velocity[i] = self.inertia_weight * velocity[i] + r1 * (global_best_position - position[i])
                position[i] += velocity[i]
                position[i] = np.clip(position[i], lb, ub)

                if np.random.rand() < self.local_perturbation_strength:
                    perturbation = np.random.normal(scale=self.quantum_scale, size=self.dim)
                    position[i] += perturbation
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

        return global_best_position, global_best_value