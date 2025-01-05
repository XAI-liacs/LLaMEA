import numpy as np
from collections import deque

class MO_QPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')
        self.swarm_size = 20
        self.swarms = []
        self.phi_base = np.pi / 6
        self.archive = deque(maxlen=50)
        self.diversity_threshold = 0.1

    def initialize_swarm(self, lb, ub):
        swarm = []
        for _ in range(self.swarm_size):
            position = lb + (ub - lb) * np.random.rand(self.dim)
            velocity = np.zeros(self.dim)
            best_position = position.copy()
            best_value = float('inf')
            swarm.append({'position': position, 'velocity': velocity, 'best_position': best_position, 'best_value': best_value})
        return swarm

    def quantum_update(self, particle, global_best, lb, ub):
        for i in range(self.dim):
            r = np.random.rand()
            progress = len(self.archive) / self.archive.maxlen
            phi = self.phi_base * (1 + np.cos(np.pi * progress))
            theta = phi if r < 0.5 else -phi
            particle['velocity'][i] = particle['velocity'][i] * np.cos(theta) + (global_best[i] - particle['position'][i]) * np.sin(theta)
            particle['position'][i] += particle['velocity'][i]
            
            if particle['position'][i] < lb[i] or particle['position'][i] > ub[i]:
                particle['position'][i] = lb[i] + (ub[i] - lb[i]) * np.random.rand()

        particle['position'] = np.clip(particle['position'], lb, ub)

    def update_archive(self, particle):
        if len(self.archive) < self.archive.maxlen or any(np.linalg.norm(particle['position'] - archived['position']) > self.diversity_threshold for archived in self.archive):
            self.archive.append(particle)

    def pareto_dominance(self, a, b):
        return all(a <= b) and any(a < b)

    def select_global_best(self):
        if not self.archive:
            return None
        
        non_dominated = [self.archive[0]]
        for candidate in self.archive:
            if any(self.pareto_dominance(candidate['best_value'], other['best_value']) for other in non_dominated):
                continue
            non_dominated = [c for c in non_dominated if not self.pareto_dominance(c['best_value'], candidate['best_value'])]
            non_dominated.append(candidate)

        return non_dominated[np.random.choice(len(non_dominated))]['best_position']

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
                
                self.update_archive(particle)
                
                if value < self.best_value:
                    self.best_value = value
                    self.best_solution = particle['position'].copy()

                if evaluations >= self.budget:
                    break

            global_best = self.select_global_best()
            for particle_index, particle in enumerate(self.swarms):
                if global_best is not None:
                    self.quantum_update(particle, global_best, lb, ub)

        return self.best_solution, self.best_value