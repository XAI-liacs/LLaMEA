import numpy as np

class CooperativeParticleDynamics:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = np.inf
        self.particle_count = 5
        self.speeds = np.zeros((self.particle_count, self.dim))
        self.positions = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.particle_count, self.dim))
        values = np.array([func(pos) for pos in self.positions])
        
        best_local_positions = np.copy(self.positions)
        best_local_values = np.copy(values)
        
        global_best_position = self.positions[np.argmin(values)]
        global_best_value = np.min(values)

        for _ in range(self.budget - self.particle_count):
            for i in range(self.particle_count):
                inertia = 0.5 + np.random.rand() / 2
                cognitive_term = np.random.rand(self.dim) * (best_local_positions[i] - self.positions[i])
                social_term = np.random.rand(self.dim) * (global_best_position - self.positions[i])
                
                self.speeds[i] = inertia * self.speeds[i] + cognitive_term + social_term
                self.positions[i] += self.speeds[i]
                self.positions[i] = np.clip(self.positions[i], lb, ub)
                
                current_value = func(self.positions[i])
                
                if current_value < best_local_values[i]:
                    best_local_positions[i] = self.positions[i]
                    best_local_values[i] = current_value
                
                if current_value < global_best_value:
                    global_best_position = self.positions[i]
                    global_best_value = current_value
        
        self.best_solution = global_best_position
        self.best_value = global_best_value

        return self.best_solution, self.best_value