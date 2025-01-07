import numpy as np

class RefinedCooperativePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(50, budget // 10)
        self.best_global_position = None
        self.best_global_fitness = float('inf')
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.quantum_coeff = 0.05  # Quantum-inspired coefficient for perturbation

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        velocity = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        pop_position = lb + (ub - lb) * np.random.rand(self.pop_size, self.dim)
        personal_best_positions = np.copy(pop_position)
        personal_best_fitness = np.full(self.pop_size, float('inf'))

        evaluations = 0
        turbulence_intensity = 0.1

        while evaluations < self.budget:
            # Divide swarm into subgroups dynamically based on convergence behavior
            subgroups = self._divide_swarm(pop_position, num_groups=2)

            for group in subgroups:
                for i in group:
                    fitness = func(pop_position[i])
                    evaluations += 1

                    if fitness < personal_best_fitness[i]:
                        personal_best_fitness[i] = fitness
                        personal_best_positions[i] = pop_position[i]

                    if fitness < self.best_global_fitness:
                        self.best_global_fitness = fitness
                        self.best_global_position = pop_position[i]

                    if evaluations >= self.budget:
                        break

            for group in subgroups:
                for i in group:
                    r1 = np.random.rand(self.dim)
                    r2 = np.random.rand(self.dim)

                    velocity[i] = (self.inertia_weight * velocity[i] +
                                   self.cognitive_coeff * r1 * (personal_best_positions[i] - pop_position[i]) +
                                   self.social_coeff * r2 * (self.best_global_position - pop_position[i]))

                    if np.random.rand() < turbulence_intensity:
                        velocity[i] += np.random.normal(0, 0.1, self.dim)

                    pop_position[i] = pop_position[i] + velocity[i]
                    pop_position[i] = np.clip(pop_position[i], lb, ub)

                    # Quantum-inspired perturbation to enhance exploration
                    if evaluations % (self.budget // 4) == 0:
                        pop_position[i] += self.quantum_coeff * np.random.normal(0, 1, self.dim)
                        pop_position[i] = np.clip(pop_position[i], lb, ub)

            turbulence_intensity = max(0.01, turbulence_intensity * 0.99)

        return self.best_global_position, self.best_global_fitness

    def _divide_swarm(self, positions, num_groups=2):
        distances = np.linalg.norm(positions - self.best_global_position, axis=1)
        sorted_indices = np.argsort(distances)
        return np.array_split(sorted_indices, num_groups)