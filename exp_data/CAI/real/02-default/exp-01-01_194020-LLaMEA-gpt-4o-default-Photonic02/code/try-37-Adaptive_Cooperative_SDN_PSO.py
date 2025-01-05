import numpy as np

class Adaptive_Cooperative_SDN_PSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')
        self.swarm_size = 20
        self.swarms = []
        self.inertia_max = 0.9
        self.inertia_min = 0.4
        self.neighborhood_factor = 0.25  # factor to dynamically adjust neighborhood size

    def initialize_swarm(self, lb, ub):
        swarm = []
        for _ in range(self.swarm_size):
            position = lb + (ub - lb) * np.random.rand(self.dim)
            velocity = np.random.rand(self.dim) * (ub - lb) * 0.1
            swarm.append({'position': position, 'velocity': velocity, 'best_position': position, 'best_value': float('inf')})
        return swarm

    def dynamic_inertia(self, evaluations):
        return self.inertia_max - ((self.inertia_max - self.inertia_min) * (evaluations / self.budget))

    def update_particle(self, particle, neighborhood_best, lb, ub, inertia_weight):
        r1, r2 = np.random.rand(), np.random.rand()
        cognitive_component = r1 * (particle['best_position'] - particle['position'])
        social_component = r2 * (neighborhood_best - particle['position'])
        particle['velocity'] = inertia_weight * particle['velocity'] + cognitive_component + social_component
        particle['position'] += particle['velocity']
        particle['position'] = np.clip(particle['position'], lb, ub)

    def adaptive_neighborhood_best(self, particle_index, evaluations):
        neighborhood_size = max(2, int(self.swarm_size * self.neighborhood_factor * (1 - evaluations / self.budget)))
        neighborhood_indices = np.random.choice(self.swarm_size, neighborhood_size, replace=False)
        neighborhood_best = self.swarms[neighborhood_indices[0]]['best_position']
        neighborhood_best_value = self.swarms[neighborhood_indices[0]]['best_value']
        
        for idx in neighborhood_indices:
            if self.swarms[idx]['best_value'] < neighborhood_best_value:
                neighborhood_best = self.swarms[idx]['best_position']
                neighborhood_best_value = self.swarms[idx]['best_value']
        
        return neighborhood_best

    def cooperative_learning(self, particle_index):
        partners = np.random.choice([i for i in range(self.swarm_size) if i != particle_index], 2, replace=False)
        knowledge = (self.swarms[partners[0]]['best_position'] + self.swarms[partners[1]]['best_position']) / 2
        return knowledge

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        self.swarms = self.initialize_swarm(lb, ub)
        
        while evaluations < self.budget:
            for particle_index, particle in enumerate(self.swarms):
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
            
            for particle_index, particle in enumerate(self.swarms):
                neighborhood_best = self.adaptive_neighborhood_best(particle_index, evaluations)
                cooperative_knowledge = self.cooperative_learning(particle_index)
                combined_best = (neighborhood_best + cooperative_knowledge) / 2
                self.update_particle(particle, combined_best, lb, ub, inertia_weight)

        return self.best_solution, self.best_value