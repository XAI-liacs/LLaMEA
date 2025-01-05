import numpy as np

class QE_ANPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')
        self.swarm_size = 20
        self.swarms = []
        self.phi_base = np.pi / 4
        self.phi_variable = np.pi / 8
        self.neighborhood_size = 5

    def initialize_swarm(self, lb, ub):
        swarm = []
        for _ in range(self.swarm_size):
            position = lb + (ub - lb) * np.random.rand(self.dim)
            velocity = np.zeros(self.dim)
            swarm.append({'position': position, 'velocity': velocity, 'best_position': position, 'best_value': float('inf')})
        return swarm

    def adaptive_quantum_update(self, particle, global_best, lb, ub, evaluations):
        for i in range(self.dim):
            r = np.random.rand()
            progress = evaluations / self.budget
            phi = self.phi_base + self.phi_variable * np.sin(np.pi * progress)
            theta = phi if r < 0.5 else -phi
            particle['velocity'][i] = particle['velocity'][i] * np.cos(theta) + (global_best[i] - particle['position'][i]) * np.sin(theta)
            particle['position'][i] += particle['velocity'][i]
            if particle['position'][i] < lb[i] or particle['position'][i] > ub[i]:
                particle['position'][i] = lb[i] + (ub[i] - lb[i]) * np.random.rand()

        particle['position'] = np.clip(particle['position'], lb, ub)

    def neighborhood_best(self, particle_index):
        neighborhood_indices = np.random.choice(self.swarm_size, self.neighborhood_size, replace=False)
        neighborhood_best = self.swarms[particle_index]['best_position']
        neighborhood_best_value = self.swarms[particle_index]['best_value']
        
        for idx in neighborhood_indices:
            if self.swarms[idx]['best_value'] < neighborhood_best_value:
                neighborhood_best = self.swarms[idx]['best_position']
                neighborhood_best_value = self.swarms[idx]['best_value']
        
        return neighborhood_best

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        self.swarms = self.initialize_swarm(lb, ub)
        
        while evaluations < self.budget:
            for particle_index, particle in enumerate(self.swarms):
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

            for particle_index, particle in enumerate(self.swarms):
                neighborhood_best = self.neighborhood_best(particle_index)
                self.adaptive_quantum_update(particle, neighborhood_best, lb, ub, evaluations)

        return self.best_solution, self.best_value