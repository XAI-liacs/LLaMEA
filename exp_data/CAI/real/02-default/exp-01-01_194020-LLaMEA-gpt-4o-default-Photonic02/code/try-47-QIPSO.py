import numpy as np

class QIPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')
        self.swarm_size = 20
        self.alpha = 0.1  # Quantum rotation angle
        self.swarms = []

    def initialize_swarm(self, lb, ub):
        swarm = []
        for _ in range(self.swarm_size):
            position = lb + (ub - lb) * np.random.rand(self.dim)
            velocity = np.random.rand(self.dim) * (ub - lb) * 0.1
            q_state = np.array([np.exp(1j * self.alpha * np.random.rand()) for _ in range(self.dim)])
            swarm.append({'position': position, 'velocity': velocity, 'best_position': position, 'best_value': float('inf'), 'q_state': q_state})
        return swarm

    def quantum_superposition(self, particle, lb, ub):
        new_position = lb + (ub - lb) * np.abs(particle['q_state'])
        return np.clip(new_position, lb, ub)

    def update_particle(self, particle, global_best, lb, ub):
        r1, r2 = np.random.rand(), np.random.rand()
        cognitive_component = r1 * (particle['best_position'] - particle['position'])
        social_component = r2 * (global_best - particle['position'])
        quantum_component = self.quantum_superposition(particle, lb, ub) - particle['position']
        particle['velocity'] = cognitive_component + social_component + quantum_component
        particle['position'] += particle['velocity']
        particle['position'] = np.clip(particle['position'], lb, ub)
        particle['q_state'] = np.exp(1j * self.alpha * np.random.rand(self.dim))

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

            global_best = min(self.swarms, key=lambda p: p['best_value'])['best_position']
            
            for particle in self.swarms:
                self.update_particle(particle, global_best, lb, ub)

        return self.best_solution, self.best_value