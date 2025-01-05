import numpy as np

class AQIPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')
        self.initial_swarm_size = 20
        self.swarms = []
        self.phi_range = (np.pi / 6, np.pi / 3)  # Dynamic quantum rotation angle range
        self.contraction_factor = 0.9  # To dynamically reduce swarm size

    def initialize_swarm(self, lb, ub, swarm_size):
        swarm = []
        for _ in range(swarm_size):
            position = lb + (ub - lb) * np.random.rand(self.dim)
            velocity = np.zeros(self.dim)
            swarm.append({'position': position, 'velocity': velocity, 'best_position': position, 'best_value': float('inf')})
        return swarm

    def quantum_update(self, particle, global_best, lb, ub, phi):
        for i in range(self.dim):
            r = np.random.rand()
            theta = phi if r < 0.5 else -phi
            particle['velocity'][i] = particle['velocity'][i] * np.cos(theta) + (global_best[i] - particle['position'][i]) * np.sin(theta)
            particle['position'][i] += particle['velocity'][i]
            if particle['position'][i] < lb[i] or particle['position'][i] > ub[i]:
                particle['position'][i] = lb[i] + (ub[i] - lb[i]) * np.random.rand()

        particle['position'] = np.clip(particle['position'], lb, ub)

    def adapt_phi(self, evaluations):
        # Linearly interpolate phi based on the remaining budget
        progress = evaluations / self.budget
        return self.phi_range[0] + progress * (self.phi_range[1] - self.phi_range[0])

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        swarm_size = self.initial_swarm_size
        self.swarms = self.initialize_swarm(lb, ub, swarm_size)
        
        while evaluations < self.budget:
            phi = self.adapt_phi(evaluations)
            swarm_size = max(5, int(swarm_size * self.contraction_factor))  # Reduce swarm size over time
            
            new_swarms = self.initialize_swarm(lb, ub, swarm_size - len(self.swarms))
            self.swarms.extend(new_swarms)

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
                self.quantum_update(particle, self.best_solution, lb, ub, phi)

        return self.best_solution, self.best_value