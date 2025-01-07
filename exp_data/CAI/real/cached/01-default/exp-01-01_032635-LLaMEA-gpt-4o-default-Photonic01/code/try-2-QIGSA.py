import numpy as np

class QIGSA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.positions = None
        self.velocities = None
        self.masses = None
        self.global_best_position = None
        self.global_best_value = np.inf
        self.iteration = 0

    def initialize_agents(self, lb, ub):
        self.positions = np.random.uniform(lb, ub, (self.budget, self.dim))
        self.velocities = np.zeros((self.budget, self.dim))
        self.masses = np.ones(self.budget)

    def update_masses(self, values):
        best_value = np.min(values)
        worst_value = np.max(values)
        if best_value == worst_value:
            self.masses = np.ones(self.budget)
        else:
            self.masses = (values - worst_value) / (best_value - worst_value)
        self.masses /= np.sum(self.masses)

    def quantum_gravitational_attraction(self, position, mass, global_best):
        direction = np.random.uniform(-1, 1, self.dim)
        attraction = np.abs(global_best - position) * np.random.rand(self.dim)
        gravitational_effect = mass * direction * attraction
        return gravitational_effect

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_agents(lb, ub)

        while self.iteration < self.budget:
            values = np.array([func(self.positions[i]) for i in range(self.budget)])
            for i in range(self.budget):
                # Update global best
                if values[i] < self.global_best_value:
                    self.global_best_value = values[i]
                    self.global_best_position = self.positions[i].copy()

            # Update masses based on current values
            self.update_masses(values)

            # Update velocities and positions
            for i in range(self.budget):
                gravitational_force = self.quantum_gravitational_attraction(
                    self.positions[i], self.masses[i], self.global_best_position
                )
                self.velocities[i] = gravitational_force
                self.positions[i] = np.clip(self.positions[i] + self.velocities[i], lb, ub)

            self.iteration += 1

        return self.global_best_position, self.global_best_value