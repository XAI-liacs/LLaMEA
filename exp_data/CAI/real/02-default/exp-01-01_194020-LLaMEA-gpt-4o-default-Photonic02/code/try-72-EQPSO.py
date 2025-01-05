import numpy as np

class EQPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')
        self.initial_swarm_size = 20
        self.max_swarm_size = 50
        self.swarms = []

    def initialize_swarm(self, lb, ub):
        swarm = []
        for _ in range(self.initial_swarm_size):
            position = lb + (ub - lb) * np.random.rand(self.dim)
            velocity = np.random.rand(self.dim) * (ub - lb) * 0.1
            swarm.append({'position': position, 'velocity': velocity, 'best_position': position, 'best_value': float('inf')})
        return swarm

    def update_particle(self, particle, global_best, lb, ub, beta, inertia_weight):
        r1, r2 = np.random.rand(), np.random.rand()
        mean_best = (particle['best_position'] + global_best) / 2
        phi = np.arccos(1 - 2 * np.random.rand(self.dim))
        direction = np.sign(np.random.rand(self.dim) - 0.5)
        
        particle['position'] = mean_best + beta * (r1 - 0.5) * np.abs(global_best - particle['position']) * np.tan(phi) * direction
        particle['position'] = np.clip(particle['position'], lb, ub)
        particle['velocity'] = inertia_weight * particle['velocity'] + r1 * (particle['best_position'] - particle['position']) + r2 * (global_best - particle['position'])
        particle['position'] += particle['velocity']

    def randomized_mutation(self, particle, lb, ub, evaluation_ratio):
        if np.random.rand() < 0.5 * (1 - evaluation_ratio):
            mutation_vector = (ub - lb) * (np.random.rand(self.dim) - 0.5) * 0.1
            particle['position'] += mutation_vector
            particle['position'] = np.clip(particle['position'], lb, ub)

    def dynamic_swarm_size(self, evaluation_ratio):
        return int(self.initial_swarm_size + evaluation_ratio * (self.max_swarm_size - self.initial_swarm_size))

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        self.swarms = self.initialize_swarm(lb, ub)
        global_best = None
        global_best_value = float('inf')
        
        while evaluations < self.budget:
            current_swarm_size = self.dynamic_swarm_size(evaluations / self.budget)
            if len(self.swarms) < current_swarm_size:
                for _ in range(current_swarm_size - len(self.swarms)):
                    position = lb + (ub - lb) * np.random.rand(self.dim)
                    velocity = np.random.rand(self.dim) * (ub - lb) * 0.1
                    self.swarms.append({'position': position, 'velocity': velocity, 'best_position': position, 'best_value': float('inf')})

            for particle_index, particle in enumerate(self.swarms[:current_swarm_size]):
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
            inertia_weight = 0.9 - 0.5 * (evaluations / self.budget)
            for particle_index, particle in enumerate(self.swarms[:current_swarm_size]):
                self.update_particle(particle, global_best, lb, ub, beta, inertia_weight)
                self.randomized_mutation(particle, lb, ub, evaluations / self.budget)

        return global_best, global_best_value