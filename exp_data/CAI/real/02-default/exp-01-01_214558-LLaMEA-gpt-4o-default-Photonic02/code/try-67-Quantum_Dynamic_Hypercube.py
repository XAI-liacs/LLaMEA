import numpy as np

class Quantum_Dynamic_Hypercube:
    def __init__(self, budget, dim, base_group_size=10, quantum_prob=0.2, exploration_factor=0.5):
        self.budget = budget
        self.dim = dim
        self.base_group_size = base_group_size
        self.quantum_prob = quantum_prob
        self.evaluations = 0
        self.exploration_factor = exploration_factor

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_global_position = np.random.uniform(lb, ub, self.dim)
        best_global_value = func(best_global_position)
        self.evaluations += 1

        hypercubes = self.initialize_hypercubes(self.base_group_size, lb, ub)

        while self.evaluations < self.budget:
            for i in range(self.base_group_size):
                center = hypercubes[i]['center']
                size = hypercubes[i]['size']

                if np.random.rand() < self.quantum_prob:
                    position = self.quantum_perturbation(center, size, lb, ub)
                else:
                    position = np.random.uniform(center - size / 2, center + size / 2)

                position = np.clip(position, lb, ub)
                value = func(position)
                self.evaluations += 1

                if value < best_global_value:
                    best_global_value = value
                    best_global_position = position

                if self.evaluations >= self.budget:
                    break

            self.adapt_hypercubes(hypercubes, best_global_position, lb, ub)

        return best_global_position

    def initialize_hypercubes(self, group_size, lb, ub):
        hypercubes = []
        for _ in range(group_size):
            center = np.random.uniform(lb, ub, self.dim)
            size = (ub - lb) * self.exploration_factor
            hypercubes.append({'center': center, 'size': size})
        return hypercubes

    def quantum_perturbation(self, center, size, lb, ub):
        q_position = center + np.random.uniform(-size / 2, size / 2)
        return np.clip(q_position, lb, ub)

    def adapt_hypercubes(self, hypercubes, best_global_position, lb, ub):
        for hypercube in hypercubes:
            center = hypercube['center']
            direction = best_global_position - center
            hypercube['center'] = center + direction * self.exploration_factor
            hypercube['center'] = np.clip(hypercube['center'], lb, ub)