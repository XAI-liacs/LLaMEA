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
            swarm.append({'position': position, 'velocity': np.zeros(self.dim), 'best_position': position, 'best_value': float('inf')})
        return swarm

    def update_particle(self, particle, global_best, lb, ub, beta, learning_rate):
        r1, r2 = np.random.rand(), np.random.rand()
        mean_best = (particle['best_position'] + global_best) / 2
        phi = np.pi * (r1 - 0.5)
        direction = np.sign(np.random.rand(self.dim) - 0.5)
        
        particle['position'] = mean_best + beta * learning_rate * np.abs(global_best - particle['position']) * np.tan(phi) * direction
        particle['position'] = np.clip(particle['position'], lb, ub)

    def randomized_mutation(self, particle, lb, ub, diversity):
        if np.random.rand() < 0.5 * (1 - diversity):
            mutation_vector = (ub - lb) * (np.random.rand(self.dim) - 0.5) * 0.1
            particle['position'] += mutation_vector
            particle['position'] = np.clip(particle['position'], lb, ub)

    def calculate_diversity(self, swarm):
        positions = np.array([p['position'] for p in swarm])
        centroid = np.mean(positions, axis=0)
        diversity = np.mean(np.linalg.norm(positions - centroid, axis=1))
        return diversity

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        self.swarms = self.initialize_swarm(lb, ub)
        global_best = None
        global_best_value = float('inf')
        elite_fraction = 0.2
        learning_rate = 0.1
        
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
            diversity = self.calculate_diversity(self.swarms)
            learning_rate = 0.5 * (1 + diversity)
            
            elite_particles = sorted(self.swarms, key=lambda p: p['best_value'])[:int(self.swarm_size * elite_fraction)]
            for particle in self.swarms:
                self.update_particle(particle, global_best, lb, ub, beta, learning_rate)
                self.randomized_mutation(particle, lb, ub, diversity)
                if particle in elite_particles:
                    particle['position'] = particle['best_position'].copy()

        return global_best, global_best_value