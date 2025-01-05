import numpy as np

class AQPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')
        self.swarm_size = 20
        self.swarms = []
        self.inertia_max = 0.9
        self.inertia_min = 0.4
        self.neighborhood_size = 5
        self.alpha = 0.5  # quantum factor

    def initialize_swarm(self, lb, ub):
        swarm = []
        for _ in range(self.swarm_size):
            position = lb + (ub - lb) * np.random.rand(self.dim)
            velocity = np.random.rand(self.dim) * (ub - lb) * 0.1
            swarm.append({'position': position, 'velocity': velocity, 'best_position': position, 'best_value': float('inf')})
        return swarm

    def dynamic_inertia(self, evaluations):
        return self.inertia_max - ((self.inertia_max - self.inertia_min) * (evaluations / self.budget))

    def update_particle(self, particle, global_best, lb, ub, inertia_weight):
        r1, r2, r3 = np.random.rand(), np.random.rand(), np.random.rand()
        cognitive_component = r1 * (particle['best_position'] - particle['position'])
        social_component = r2 * (global_best - particle['position'])
        quantum_component = r3 * (self.alpha * (global_best + particle['best_position']) / 2 - particle['position'])
        particle['velocity'] = inertia_weight * particle['velocity'] + cognitive_component + social_component + quantum_component
        particle['position'] += particle['velocity']
        particle['position'] = np.clip(particle['position'], lb, ub)

    def global_best(self):
        global_best = self.swarms[0]['best_position']
        global_best_value = self.swarms[0]['best_value']
        
        for particle in self.swarms:
            if particle['best_value'] < global_best_value:
                global_best = particle['best_position']
                global_best_value = particle['best_value']
        
        return global_best

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        self.swarms = self.initialize_swarm(lb, ub)
        
        while evaluations < self.budget:
            global_best = self.global_best()

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

            inertia_weight = self.dynamic_inertia(evaluations)
            
            for particle in self.swarms:
                self.update_particle(particle, global_best, lb, ub, inertia_weight)

        return self.best_solution, self.best_value