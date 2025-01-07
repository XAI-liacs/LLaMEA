import numpy as np

class AQSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')
        self.swarm_size = 20
        self.swarms = []
        self.alpha = 0.5
        self.beta = 0.5

    def initialize_swarm(self, lb, ub):
        swarm = []
        for _ in range(self.swarm_size):
            position = lb + (ub - lb) * np.random.rand(self.dim)
            quantum_prob = np.ones(self.dim) * 0.5
            swarm.append({'position': position, 'quantum_prob': quantum_prob, 'best_position': position, 'best_value': float('inf')})
        return swarm

    def adaptive_parameters(self, evaluations):
        progress_ratio = evaluations / self.budget
        self.alpha = max(0.4, 0.9 * (1 - progress_ratio))
        self.beta = min(0.6, 0.1 + 0.5 * progress_ratio)

    def quantum_sample(self, particle, lb, ub):
        particle['position'] = lb + (ub - lb) * np.where(np.random.rand(self.dim) < particle['quantum_prob'], 1, 0)
        particle['position'] = np.clip(particle['position'], lb, ub)

    def update_particle(self, particle, global_best, lb, ub):
        r1, r2 = np.random.rand(), np.random.rand()
        particle['quantum_prob'] = (self.alpha * particle['quantum_prob'] +
                                    self.beta * (global_best - particle['position']))
        self.quantum_sample(particle, lb, ub)

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

            self.adaptive_parameters(evaluations)
            global_best = min(self.swarms, key=lambda p: p['best_value'])['best_position']
            
            for particle in self.swarms:
                self.update_particle(particle, global_best, lb, ub)

        return self.best_solution, self.best_value