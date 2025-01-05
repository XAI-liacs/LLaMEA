import numpy as np

class AQASO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')
        self.swarm_size = 20
        self.swarms = []

    def initialize_swarm(self, lb, ub):
        swarm = []
        for _ in range(self.swarm_size):
            position = lb + (ub - lb) * np.random.rand(self.dim)
            velocity = np.random.rand(self.dim) * (ub - lb) * 0.1
            swarm.append({'position': position, 'velocity': velocity, 'best_position': position, 'best_value': float('inf')})
        return swarm

    def update_particle(self, particle, global_best, lb, ub, temperature):
        r = np.random.rand(self.dim)
        quantum_factor = np.tanh(temperature * (global_best - particle['position']))
        particle['position'] += r * quantum_factor * (global_best - particle['position'])
        particle['position'] = np.clip(particle['position'], lb, ub)

    def adaptive_mutation(self, particle, lb, ub, evaluation_ratio):
        mutation_prob = 0.5 * (1 - evaluation_ratio)
        if np.random.rand() < mutation_prob:
            mutation_scale = (ub - lb) * 0.1 * np.exp(-20 * evaluation_ratio)
            mutation_vector = np.random.normal(0, mutation_scale, self.dim)
            particle['position'] += mutation_vector
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

            temperature = 1.0 - evaluations / self.budget
            for particle_index, particle in enumerate(self.swarms):
                self.update_particle(particle, global_best, lb, ub, temperature)
                self.adaptive_mutation(particle, lb, ub, evaluations / self.budget)

        return global_best, global_best_value