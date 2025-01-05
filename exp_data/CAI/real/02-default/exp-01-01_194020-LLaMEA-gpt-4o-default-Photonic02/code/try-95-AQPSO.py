import numpy as np

class AQPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')
        self.swarm_size = 25  # Increased swarm size for diversified exploration
        self.swarms = []

    def initialize_swarm(self, lb, ub):
        swarm = []
        for _ in range(self.swarm_size):
            position = lb + (ub - lb) * np.random.rand(self.dim)
            swarm.append({'position': position, 'best_position': position, 'best_value': float('inf')})
        return swarm

    def update_particle(self, particle, global_best, lb, ub, beta):
        r1 = np.random.rand(self.dim)
        direction = np.sign(np.random.rand(self.dim) - 0.5)
        mean_best = (particle['best_position'] + global_best) / 2
        
        phi = np.arccos(1 - 2 * np.random.rand(self.dim))
        adaptive_factor = np.exp(-beta * np.abs(global_best - particle['position']))
        particle['position'] = mean_best + beta * adaptive_factor * np.tan(phi) * direction
        particle['position'] = np.clip(particle['position'], lb, ub)

    def adaptive_mutation(self, particle, lb, ub, evaluation_ratio):
        mutation_probability = 0.5 * (1 - evaluation_ratio)
        if np.random.rand() < mutation_probability:
            mutation_vector = (ub - lb) * (2 * np.random.rand(self.dim) - 1) * (0.1 * (1 - evaluation_ratio))
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

            beta = 1.0 - evaluations / self.budget
            for particle_index, particle in enumerate(self.swarms):
                self.update_particle(particle, global_best, lb, ub, beta)
                self.adaptive_mutation(particle, lb, ub, evaluations / self.budget)

        return global_best, global_best_value