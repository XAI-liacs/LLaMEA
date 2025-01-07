import numpy as np

class EQMS_PSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')
        self.swarm_size = 20
        self.num_swarms = 3
        self.swarms = []
        self.phi_base = np.pi / 4
        self.phi_variable = np.pi / 8
        self.neighborhood_size = 5
        self.interaction_frequency = 50

    def initialize_swarm(self, lb, ub):
        swarms = []
        for _ in range(self.num_swarms):
            swarm = []
            for _ in range(self.swarm_size):
                position = lb + (ub - lb) * np.random.rand(self.dim)
                velocity = np.zeros(self.dim)
                swarm.append({'position': position, 'velocity': velocity, 'best_position': position, 'best_value': float('inf')})
            swarms.append(swarm)
        return swarms

    def quantum_update(self, particle, global_best, local_best, lb, ub, evaluations):
        for i in range(self.dim):
            r = np.random.rand()
            progress = evaluations / self.budget
            phi = self.phi_base + self.phi_variable * np.sin(np.pi * progress)
            theta = phi if r < 0.5 else -phi
            velocity_contribution = (global_best[i] - particle['position'][i]) * np.sin(theta)
            neighborhood_contribution = (local_best[i] - particle['position'][i]) * np.sin(theta / 2)
            particle['velocity'][i] = particle['velocity'][i] * np.cos(theta) + velocity_contribution + neighborhood_contribution
            particle['position'][i] += particle['velocity'][i]
            if particle['position'][i] < lb[i] or particle['position'][i] > ub[i]:
                particle['position'][i] = lb[i] + (ub[i] - lb[i]) * np.random.rand()

        particle['position'] = np.clip(particle['position'], lb, ub)

    def neighborhood_best(self, swarm, particle_index):
        neighborhood_indices = np.random.choice(self.swarm_size, self.neighborhood_size, replace=False)
        neighborhood_best = swarm[particle_index]['best_position']
        neighborhood_best_value = swarm[particle_index]['best_value']
        
        for idx in neighborhood_indices:
            if swarm[idx]['best_value'] < neighborhood_best_value:
                neighborhood_best = swarm[idx]['best_position']
                neighborhood_best_value = swarm[idx]['best_value']
        
        return neighborhood_best

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        self.swarms = self.initialize_swarm(lb, ub)
        
        while evaluations < self.budget:
            for swarm in self.swarms:
                for particle_index, particle in enumerate(swarm):
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

                for particle_index, particle in enumerate(swarm):
                    neighborhood_best = self.neighborhood_best(swarm, particle_index)
                    self.quantum_update(particle, self.best_solution, neighborhood_best, lb, ub, evaluations)

            if evaluations % self.interaction_frequency == 0:
                self.synchronize_swarms()

        return self.best_solution, self.best_value

    def synchronize_swarms(self):
        global_best_positions = [swarms[i][np.argmin([p['best_value'] for p in swarms[i]])]['best_position'] for i in range(self.num_swarms)]
        
        for i, swarm in enumerate(self.swarms):
            for particle in swarm:
                target_swarm_index = (i + 1) % self.num_swarms
                target_position = global_best_positions[target_swarm_index]
                particle['position'] += (target_position - particle['position']) * np.random.rand()
                particle['position'] = np.clip(particle['position'], lb, ub)