import numpy as np

class QEASS:
    def __init__(self, budget, dim, num_spheres=5, points_per_sphere=8, sphere_radius=0.1, quantum_prob=0.2, cooperation_rate=0.3):
        self.budget = budget
        self.dim = dim
        self.num_spheres = num_spheres
        self.points_per_sphere = points_per_sphere
        self.sphere_radius = sphere_radius
        self.quantum_prob = quantum_prob
        self.cooperation_rate = cooperation_rate
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_global_position = None
        best_global_value = float('inf')

        spheres = [self.initialize_sphere(lb, ub) for _ in range(self.num_spheres)]
        sphere_centers = [np.mean(sphere, axis=0) for sphere in spheres]

        while self.evaluations < self.budget:
            for sphere_id in range(self.num_spheres):
                local_best_position = None
                local_best_value = float('inf')

                for i in range(self.points_per_sphere):
                    position = spheres[sphere_id][i]

                    if np.random.rand() < self.quantum_prob:
                        position = self.quantum_perturbation(position, lb, ub)

                    value = func(position)
                    self.evaluations += 1

                    if value < local_best_value:
                        local_best_value = value
                        local_best_position = position

                    if value < best_global_value:
                        best_global_value = value
                        best_global_position = position

                    if self.evaluations >= self.budget:
                        break

                if local_best_value < best_global_value:
                    best_global_value = local_best_value
                    best_global_position = local_best_position

                # Adaptive sphere expansion and contraction
                if local_best_value < best_global_value:
                    self.sphere_radius *= 1.1
                else:
                    self.sphere_radius *= 0.9

                # Cooperative adaptation
                if np.random.rand() < self.cooperation_rate:
                    self.cooperative_transfer(spheres, sphere_id)

                if self.evaluations >= self.budget:
                    break

        return best_global_position

    def initialize_sphere(self, lb, ub):
        center = np.random.uniform(lb, ub, self.dim)
        return [self.random_point_in_sphere(center, lb, ub) for _ in range(self.points_per_sphere)]

    def random_point_in_sphere(self, center, lb, ub):
        direction = np.random.normal(0, 1, self.dim)
        direction /= np.linalg.norm(direction)
        point = center + direction * self.sphere_radius * np.random.rand()
        return np.clip(point, lb, ub)

    def quantum_perturbation(self, position, lb, ub):
        scale = np.random.rand()
        q_position = position + (np.random.rand(self.dim) - 0.5) * (ub - lb) * scale
        return np.clip(q_position, lb, ub)

    def cooperative_transfer(self, spheres, current_id):
        target_id = np.random.choice([i for i in range(self.num_spheres) if i != current_id])
        exchange_point = np.random.randint(self.points_per_sphere)
        spheres[current_id][exchange_point], spheres[target_id][exchange_point] = spheres[target_id][exchange_point], spheres[current_id][exchange_point]