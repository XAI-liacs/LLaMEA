import numpy as np

class DynamicMultiSwarmPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.num_swarms = 3
        self.swarm_size = 30
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.velocity_clamp = (0.1 * (self.bounds[1] - self.bounds[0]), 0.1 * (self.bounds[1] - self.bounds[0]))
        self.current_eval = 0
    
    def __call__(self, func):
        def initialize_particle():
            position = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)
            velocity = np.random.uniform(-1, 1, self.dim)
            return {'position': position, 'velocity': velocity, 'best_position': position.copy(), 'best_value': float('inf')}
        
        swarms = [[initialize_particle() for _ in range(self.swarm_size)] for _ in range(self.num_swarms)]
        global_best_position = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)
        global_best_value = float('inf')
        
        while self.current_eval < self.budget:
            for swarm in swarms:
                for particle in swarm:
                    fitness_value = func(particle['position'])
                    self.current_eval += 1
                    
                    if fitness_value < particle['best_value']:
                        particle['best_value'] = fitness_value
                        particle['best_position'] = particle['position'].copy()

                    if fitness_value < global_best_value:
                        global_best_value = fitness_value
                        global_best_position = particle['position'].copy()

                    if self.current_eval >= self.budget:
                        break

                if self.current_eval >= self.budget:
                    break
            
            for swarm in swarms:
                for particle in swarm:
                    r1, r2 = np.random.rand(), np.random.rand()
                    cognitive_component = self.cognitive_coeff * r1 * (particle['best_position'] - particle['position'])
                    social_component = self.social_coeff * r2 * (global_best_position - particle['position'])
                    particle['velocity'] = (self.inertia_weight * particle['velocity'] +
                                            cognitive_component +
                                            social_component)

                    particle['velocity'] = np.clip(particle['velocity'], self.velocity_clamp[0], self.velocity_clamp[1])
                    particle['position'] += particle['velocity']
                    particle['position'] = np.clip(particle['position'], self.bounds[0], self.bounds[1])

            # Swarm adaptation logic
            if self.current_eval % (self.budget // (2 * self.num_swarms)) == 0:
                # Reinitializing half of the worst performing swarms
                swarms.sort(key=lambda s: min(p['best_value'] for p in s))
                for i in range(self.num_swarms // 2):
                    swarms[-(i+1)] = [initialize_particle() for _ in range(self.swarm_size)]

        return global_best_position, global_best_value