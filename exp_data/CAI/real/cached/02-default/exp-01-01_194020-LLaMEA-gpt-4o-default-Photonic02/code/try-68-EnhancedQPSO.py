import numpy as np

class EnhancedQPSO:
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

    def levy_flight(self, lb, ub):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                 (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / np.abs(v) ** (1 / beta)
        return step * (ub - lb) * 0.01

    def update_particle(self, particle, global_best, lb, ub, alpha):
        mean_best = (particle['best_position'] + global_best) / 2
        direction = np.random.choice([-1, 1], size=self.dim)
        particle['position'] = mean_best + alpha * np.random.rand(self.dim) * direction * self.levy_flight(lb, ub)
        particle['position'] = np.clip(particle['position'], lb, ub)

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        self.swarms = self.initialize_swarm(lb, ub)
        global_best = None
        global_best_value = float('inf')

        while evaluations < self.budget:
            evaluation_ratio = evaluations / self.budget
            alpha = 0.1 + 0.9 * evaluation_ratio  # Adaptive convergence pressure

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

            for particle in self.swarms:
                self.update_particle(particle, global_best, lb, ub, alpha)

        return global_best, global_best_value