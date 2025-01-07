import numpy as np

class AQPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')
        self.initial_swarm_size = 20
        self.swarms = []

    def initialize_swarm(self, lb, ub):
        swarm = []
        for _ in range(self.initial_swarm_size):
            position = lb + (ub - lb) * np.random.rand(self.dim)
            swarm.append({'position': position, 'best_position': position, 'best_value': float('inf')})
        return swarm

    def update_particle(self, particle, global_best, lb, ub, beta):
        r1, r2 = np.random.rand(), np.random.rand()
        mean_best = (particle['best_position'] + global_best) / 2
        phi = np.arccos(1 - 2 * np.random.rand(self.dim))
        direction = np.sign(np.random.rand(self.dim) - 0.5)
        
        particle['position'] = mean_best + beta * (r1 - 0.5) * np.abs(global_best - particle['position']) * np.tan(phi) * direction
        particle['position'] = np.clip(particle['position'], lb, ub)

    def adaptive_resizing(self, evaluations):
        ratio = evaluations / self.budget
        new_size = max(2, int(self.initial_swarm_size * (1 - ratio / 2)))
        if len(self.swarms) > new_size:
            self.swarms = self.swarms[:new_size]

    def quantum_tunneling(self, particle, lb, ub, evaluation_ratio):
        if np.random.rand() < 0.3 * (1 - evaluation_ratio):
            direction = np.random.normal(size=self.dim)
            step_size = (ub - lb) * 0.05
            particle['position'] += step_size * direction
            particle['position'] = np.clip(particle['position'], lb, ub)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        self.swarms = self.initialize_swarm(lb, ub)
        global_best = None
        global_best_value = float('inf')
        
        while evaluations < self.budget:
            for particle_index, particle in enumerate(self.swarms):
                value = func(particle['position'])
                evaluations += 1
                
                if value < particle['best_value']:
                    particle['best_value'] = value
                    particle['best_position'] = particle['position'].copy()
                
                if value < global_best_value:
                    global_best_value = value
                    global_best = particle['position'].copy()

                if evaluations >= self.budget:
                    break

            beta = 1.0 - evaluations / self.budget
            self.adaptive_resizing(evaluations)
            for particle_index, particle in enumerate(self.swarms):
                self.update_particle(particle, global_best, lb, ub, beta)
                self.quantum_tunneling(particle, lb, ub, evaluations / self.budget)

        return global_best, global_best_value