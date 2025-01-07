import numpy as np

class AQPSO:
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
            swarm.append({'position': position, 'best_position': position, 'best_value': float('inf')})
        return swarm

    def update_particle(self, particle, global_best, lb, ub, evaluation_ratio):
        r1, r2 = np.random.rand(), np.random.rand()
        theta = np.arccos(1 - 2 * np.random.rand(self.dim))
        direction = np.sign(np.random.rand(self.dim) - 0.5)
        
        alpha = 0.5 + 0.5 * evaluation_ratio
        adaptive_beta = (1 - evaluation_ratio) * r1 + evaluation_ratio * r2
        mean_best = alpha * particle['best_position'] + (1 - alpha) * global_best
        
        particle['position'] = mean_best + adaptive_beta * np.abs(global_best - particle['position']) * np.tan(theta) * direction
        particle['position'] = np.clip(particle['position'], lb, ub)

    def dynamic_mutation(self, particle, lb, ub, evaluation_ratio):
        if np.random.rand() < 0.5 * (1 - evaluation_ratio):
            mutation_vector = (ub - lb) * (np.random.rand(self.dim) - 0.5) * 0.1 * (1 - evaluation_ratio)
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

            evaluation_ratio = evaluations / self.budget
            for particle_index, particle in enumerate(self.swarms):
                self.update_particle(particle, global_best, lb, ub, evaluation_ratio)
                self.dynamic_mutation(particle, lb, ub, evaluation_ratio)

        return global_best, global_best_value