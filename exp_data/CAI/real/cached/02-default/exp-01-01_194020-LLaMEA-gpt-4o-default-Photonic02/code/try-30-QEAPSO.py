import numpy as np

class QEAPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')
        self.swarm_size = 20
        self.swarms = []
        self.phi_base = np.pi / 4  # Base quantum rotation angle
        self.phi_variable = np.pi / 8  # Variable component for adaptive rotation
        self.learning_factor = 0.7  # Dynamic learning factor for exploration

    def initialize_swarm(self, lb, ub):
        swarm = []
        for _ in range(self.swarm_size):
            position = lb + (ub - lb) * np.random.rand(self.dim)
            velocity = np.zeros(self.dim)
            swarm.append({'position': position, 'velocity': velocity, 'best_position': position, 'best_value': float('inf')})
        return swarm

    def adaptive_quantum_update(self, particle, global_best, lb, ub, evaluations):
        for i in range(self.dim):
            r1, r2 = np.random.rand(), np.random.rand()
            progress = evaluations / self.budget
            phi1 = self.phi_base + self.phi_variable * np.sin(np.pi * progress)
            phi2 = self.phi_base - self.phi_variable * np.sin(np.pi * progress)
            theta1 = phi1 if r1 < 0.5 else -phi1
            theta2 = phi2 if r2 < 0.5 else -phi2

            # Dual quantum rotators
            weight = self.learning_factor * (1 - progress)
            particle['velocity'][i] = (
                weight * particle['velocity'][i] * np.cos(theta1) +
                (1 - weight) * (global_best[i] - particle['position'][i]) * np.sin(theta2)
            )

            particle['position'][i] += particle['velocity'][i]

            if particle['position'][i] < lb[i] or particle['position'][i] > ub[i]:
                particle['position'][i] = lb[i] + (ub[i] - lb[i]) * np.random.rand()  # Re-initialize if out of bounds

        particle['position'] = np.clip(particle['position'], lb, ub)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        self.swarms = self.initialize_swarm(lb, ub)
        
        while evaluations < self.budget:
            for particle in self.swarms:
                value = func(particle['position'])
                evaluations += 1
                
                if value < particle['best_value']:
                    particle['best_value'] = value
                    particle['best_position'] = particle['position'].copy()
                
                if value < self.best_value:
                    self.best_value = value
                    self.best_solution = particle['position'].copy()

                if evaluations >= self.budget:
                    break

            for particle in self.swarms:
                self.adaptive_quantum_update(particle, self.best_solution, lb, ub, evaluations)

        return self.best_solution, self.best_value