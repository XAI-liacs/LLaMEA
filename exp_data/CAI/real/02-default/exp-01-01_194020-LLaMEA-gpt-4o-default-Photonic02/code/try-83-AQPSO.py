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
            velocity = np.random.rand(self.dim) * (ub - lb) * 0.1
            swarm.append({'position': position, 'velocity': velocity, 'best_position': position, 'best_value': float('inf')})
        return swarm

    def update_particle(self, particle, global_best, lb, ub, beta, alpha):
        r1, r2 = np.random.rand(), np.random.rand()
        mean_best = (particle['best_position'] + global_best) / 2
        phi = np.arccos(1 - 2 * np.random.rand(self.dim))
        direction = np.sign(np.random.rand(self.dim) - 0.5)
        
        particle['position'] = mean_best + beta * np.exp(-alpha * r1) * np.abs(global_best - particle['position']) * np.tan(phi) * direction
        particle['position'] = np.clip(particle['position'], lb, ub)

    def chaotic_mutation(self, particle, lb, ub, evaluation_ratio):
        if np.random.rand() < 0.5 * (1 - evaluation_ratio):
            chaotic_sequence = (np.sin(evaluation_ratio * np.pi / 2) * 0.5 + 0.5)
            mutation_vector = (ub - lb) * (np.random.rand(self.dim) - 0.5) * chaotic_sequence * 0.1
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
            alpha = 4.0 * evaluations / self.budget  # Adaptive learning factor
            for particle_index, particle in enumerate(self.swarms):
                self.update_particle(particle, global_best, lb, ub, beta, alpha)
                self.chaotic_mutation(particle, lb, ub, evaluations / self.budget)

        return global_best, global_best_value