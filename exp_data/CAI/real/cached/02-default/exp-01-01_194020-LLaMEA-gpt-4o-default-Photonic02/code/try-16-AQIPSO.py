import numpy as np

class AQIPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')
        self.initial_swarm_size = 20
        self.swarm_size = self.initial_swarm_size
        self.phi = np.pi / 4  # Initial quantum rotation angle
        self.swarms = []

    def initialize_swarm(self, lb, ub):
        swarm = []
        for _ in range(self.swarm_size):
            position = lb + (ub - lb) * np.random.rand(self.dim)
            velocity = np.zeros(self.dim)
            swarm.append({'position': position, 'velocity': velocity, 'best_position': position, 'best_value': float('inf')})
        return swarm

    def quantum_update(self, particle, global_best, lb, ub):
        for i in range(self.dim):
            r = np.random.rand()
            theta = self.phi if r < 0.5 else -self.phi
            particle['velocity'][i] = particle['velocity'][i] * np.cos(theta) + (global_best[i] - particle['position'][i]) * np.sin(theta)
            particle['position'][i] += particle['velocity'][i]
            if particle['position'][i] < lb[i] or particle['position'][i] > ub[i]:
                particle['position'][i] = lb[i] + (ub[i] - lb[i]) * np.random.rand()

        particle['position'] = np.clip(particle['position'], lb, ub)

    def adapt_parameters(self, improvement_rate):
        # Adjust phi to increase exploration when improvements slow down
        self.phi = max(0.1, self.phi * (1 + 0.1 * (1 - improvement_rate)))
        # Adjust swarm size to enhance exploitation when improvements are good
        self.swarm_size = max(self.initial_swarm_size, int(self.swarm_size * (1 + 0.05 * improvement_rate)))

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        self.swarms = self.initialize_swarm(lb, ub)
        previous_best_value = self.best_value

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
                    improvement_rate = (previous_best_value - self.best_value) / abs(previous_best_value)
                    self.adapt_parameters(improvement_rate)
                    previous_best_value = self.best_value

                if evaluations >= self.budget:
                    break

            # Update positions based on global best
            for particle in self.swarms:
                self.quantum_update(particle, self.best_solution, lb, ub)

        return self.best_solution, self.best_value