import numpy as np

class HSEA_QE:
    def __init__(self, budget, dim, swarm_size=20, inertia=0.5, cognitive=1.5, social=1.5, mutation_rate=0.1, quantum_prob=0.2):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.mutation_rate = mutation_rate
        self.quantum_prob = quantum_prob
        self.evaluations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_global_position = None
        best_global_value = float('inf')

        positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_values = np.full(self.swarm_size, np.inf)

        while self.evaluations < self.budget:
            for i in range(self.swarm_size):
                if np.random.rand() < self.quantum_prob:
                    positions[i] = self.quantum_perturbation(positions[i], lb, ub)

                velocities[i] = (self.inertia * velocities[i] +
                                 self.cognitive * np.random.random(self.dim) * (personal_best_positions[i] - positions[i]) +
                                 self.social * np.random.random(self.dim) * (best_global_position - positions[i] if best_global_position is not None else 0))
                positions[i] = np.clip(positions[i] + velocities[i], lb, ub)

                value = func(positions[i])
                self.evaluations += 1

                if value < personal_best_values[i]:
                    personal_best_values[i] = value
                    personal_best_positions[i] = positions[i].copy()

                if value < best_global_value:
                    best_global_value = value
                    best_global_position = positions[i].copy()

                if self.evaluations >= self.budget:
                    break

            self.evolve_population(positions, lb, ub)

        return best_global_position

    def quantum_perturbation(self, position, lb, ub):
        q_position = position + (np.random.rand(self.dim) - 0.5) * (ub - lb) * 0.1
        return np.clip(q_position, lb, ub)

    def evolve_population(self, positions, lb, ub):
        for i in range(0, self.swarm_size, 2):
            if i + 1 < self.swarm_size:
                crossover_point = np.random.randint(1, self.dim)
                if np.random.rand() < 0.5:
                    positions[i][:crossover_point], positions[i + 1][:crossover_point] = (
                        positions[i + 1][:crossover_point].copy(), positions[i][:crossover_point].copy())

                if np.random.rand() < self.mutation_rate:
                    positions[i] = self.mutate(positions[i], lb, ub)
                    positions[i + 1] = self.mutate(positions[i + 1], lb, ub)

    def mutate(self, position, lb, ub):
        mutation = (np.random.rand(self.dim) - 0.5) * 0.05 * (ub - lb)
        return np.clip(position + mutation, lb, ub)