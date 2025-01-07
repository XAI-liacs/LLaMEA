import numpy as np

class PIQSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')
        self.swarm_size = 20
        self.num_swarms = 5
        self.swarms = []
        self.global_bests = []

    def initialize_swarm(self, lb, ub):
        swarm = []
        for _ in range(self.swarm_size):
            position = lb + (ub - lb) * np.random.rand(self.dim)
            swarm.append({'position': position, 'best_position': position, 'best_value': float('inf')})
        return swarm

    def update_particle_position(self, particle, local_best, lb, ub, exploration_factor):
        r1 = np.random.rand(self.dim)
        direction = np.sign(np.random.rand(self.dim) - 0.5)
        particle['position'] = local_best + exploration_factor * (r1 - 0.5) * np.abs(local_best - particle['position']) * direction
        particle['position'] = np.clip(particle['position'], lb, ub)

    def quantum_tunneling(self, particle, lb, ub, global_best_value):
        if np.random.rand() < 0.1:
            q_jump = np.random.normal(0, 1, self.dim) * (ub - lb) * 0.1
            particle['position'] += q_jump
            particle['position'] = np.clip(particle['position'], lb, ub)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        self.swarms = [self.initialize_swarm(lb, ub) for _ in range(self.num_swarms)]
        self.global_bests = [np.full(self.dim, np.inf) for _ in range(self.num_swarms)]
        global_best_value = float('inf')

        while evaluations < self.budget:
            for swarm_index, swarm in enumerate(self.swarms):
                for particle_index, particle in enumerate(swarm):
                    value = func(particle['position'])
                    evaluations += 1

                    if value < particle['best_value']:
                        particle['best_value'] = value
                        particle['best_position'] = particle['position'].copy()

                    if value < self.global_bests[swarm_index].get('value', float('inf')):
                        self.global_bests[swarm_index] = {'position': particle['position'].copy(), 'value': value}

                    if value < global_best_value:
                        global_best_value = value
                        self.best_solution = particle['position'].copy()

                    if evaluations >= self.budget:
                        break

                exploration_factor = 1.0 - evaluations / self.budget
                for particle_index, particle in enumerate(swarm):
                    self.update_particle_position(particle, self.global_bests[swarm_index]['position'], lb, ub, exploration_factor)
                    self.quantum_tunneling(particle, lb, ub, global_best_value)

            if evaluations >= self.budget:
                break

        return self.best_solution, global_best_value