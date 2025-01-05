import numpy as np

class AQPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')
        self.swarm_size = 20
        self.swarms = []
        self.alpha = 0.5  # Learning factor
        self.beta = 0.5   # Learning factor

    def initialize_swarm(self, lb, ub):
        swarm = []
        for _ in range(self.swarm_size):
            position = lb + (ub - lb) * np.random.rand(self.dim)
            velocity = np.random.rand(self.dim) * (ub - lb) * 0.1
            pbest_position = position.copy()
            pbest_value = float('inf')
            swarm.append({'position': position, 'velocity': velocity, 
                          'pbest_position': pbest_position, 'pbest_value': pbest_value})
        return swarm

    def update_particle(self, particle, gbest_position, lb, ub):
        r1, r2 = np.random.rand(), np.random.rand()
        self.alpha = 0.5 + 0.5 * np.random.rand()
        self.beta = 0.5 + 0.5 * np.random.rand()
        particle['velocity'] = self.alpha * (particle['velocity'] + 
                                             self.beta * (particle['pbest_position'] - particle['position']) +
                                             (gbest_position - particle['position']))
        particle['position'] = np.clip(particle['position'] + particle['velocity'], lb, ub)

    def quantum_behaviour(self, particle, gbest_position):
        phi = np.random.rand(self.dim)
        u = np.random.uniform(0, 1, self.dim) < 0.5
        particle['position'] = np.where(u, particle['position'] + phi * (gbest_position - particle['position']),
                                        particle['position'] - phi * (gbest_position - particle['position']))

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        self.swarms = self.initialize_swarm(lb, ub)
        gbest_position = None
        
        while evaluations < self.budget:
            for particle in self.swarms:
                value = func(particle['position'])
                evaluations += 1
                
                if value < particle['pbest_value']:
                    particle['pbest_value'] = value
                    particle['pbest_position'] = particle['position'].copy()
                
                if value < self.best_value:
                    self.best_value = value
                    self.best_solution = particle['position'].copy()
                    gbest_position = self.best_solution

                if evaluations >= self.budget:
                    break

            for particle in self.swarms:
                self.update_particle(particle, gbest_position, lb, ub)
                self.quantum_behaviour(particle, gbest_position)

        return self.best_solution, self.best_value