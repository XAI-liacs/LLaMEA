import numpy as np

class EQPSO:
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
            swarm.append({
                'position': position,
                'best_position': position,
                'best_value': float('inf')
            })
        return swarm

    def update_particle(self, particle, global_best, lb, ub, beta, gamma):
        r1 = np.random.rand()
        mean_best = (particle['best_position'] + global_best) / 2
        phi = np.arccos(1 - 2 * np.random.rand(self.dim))
        direction = np.sign(np.random.rand(self.dim) - 0.5)

        adaptive_step = (1 + gamma * np.random.rand()) * np.abs(global_best - particle['position'])
        particle['position'] = mean_best + beta * (r1 - 0.5) * adaptive_step * np.tan(phi) * direction
        particle['position'] = np.clip(particle['position'], lb, ub)

    def randomized_mutation(self, particle, lb, ub, evaluation_ratio):
        if np.random.rand() < 0.5 * (1 - evaluation_ratio**2):
            mutation_strength = 0.1 * (1 - evaluation_ratio)
            mutation_vector = (ub - lb) * (np.random.rand(self.dim) - 0.5) * mutation_strength
            particle['position'] += mutation_vector
            particle['position'] = np.clip(particle['position'], lb, ub)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        self.swarms = self.initialize_swarm(lb, ub)
        global_best = None
        global_best_value = float('inf')
        
        while evaluations < self.budget:
            for particle in self.swarms:
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
            gamma = 0.5 * (1 - evaluations / self.budget)
            for particle in self.swarms:
                self.update_particle(particle, global_best, lb, ub, beta, gamma)
                self.randomized_mutation(particle, lb, ub, evaluations / self.budget)

        return global_best, global_best_value