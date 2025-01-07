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
            velocity = np.random.rand(self.dim) * (ub - lb) * 0.1
            swarm.append({'position': position, 'velocity': velocity, 'best_position': position, 'best_value': float('inf')})
        return swarm

    def levy_flight(self, size, alpha=1.5):
        # Using Mantegna's algorithm for Lévy flight
        sigma = (np.gamma(1 + alpha) * np.sin(np.pi * alpha / 2) / 
                 (np.gamma((1 + alpha) / 2) * alpha * 2**((alpha - 1) / 2)))**(1 / alpha)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        return u / np.abs(v)**(1 / alpha)

    def update_particle(self, particle, global_best, lb, ub, beta, gamma):
        r1, r2 = np.random.rand(), np.random.rand()
        mean_best = (particle['best_position'] + global_best) / 2
        phi = np.arccos(1 - 2 * np.random.rand(self.dim))
        direction = np.sign(np.random.rand(self.dim) - 0.5)
        
        particle['position'] = mean_best + beta * (r1 - 0.5) * np.abs(global_best - particle['position']) * np.tan(phi) * direction
        particle['position'] = np.clip(particle['position'], lb, ub)
        
        if np.random.rand() < gamma:
            levy_step = self.levy_flight(self.dim)
            particle['position'] += levy_step * (ub - lb) * 0.1
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
            gamma = 0.1 * (1 + np.cos(np.pi * evaluations / self.budget))  # Adaptive mutation probability

            for particle_index, particle in enumerate(self.swarms):
                self.update_particle(particle, global_best, lb, ub, beta, gamma)

        return global_best, global_best_value