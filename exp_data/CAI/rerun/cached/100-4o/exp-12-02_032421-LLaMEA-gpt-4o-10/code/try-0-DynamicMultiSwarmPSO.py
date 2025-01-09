import numpy as np

class DynamicMultiSwarmPSO:
    def __init__(self, budget, dim, swarm_size=20, num_swarms=3):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.num_swarms = num_swarms
        self.global_best_pos = np.random.uniform(-5.0, 5.0, dim)
        self.global_best_val = np.inf
        self.inertia_weight = 0.9
        self.cognitive_coeff = 2.0
        self.social_coeff = 2.0

    def __call__(self, func):
        # Initialize swarms
        swarms = [self.initialize_swarm() for _ in range(self.num_swarms)]
        evaluations = 0

        while evaluations < self.budget:
            for swarm in swarms:
                for particle in swarm:
                    if evaluations >= self.budget:
                        break
                    # Update particle velocity
                    inertia = self.inertia_weight * particle['velocity']
                    cognitive_component = self.cognitive_coeff * np.random.rand(self.dim) * (particle['best_pos'] - particle['position'])
                    social_component = self.social_coeff * np.random.rand(self.dim) * (self.global_best_pos - particle['position'])
                    particle['velocity'] = inertia + cognitive_component + social_component
                    
                    # Update particle position
                    particle['position'] += particle['velocity']
                    particle['position'] = np.clip(particle['position'], -5.0, 5.0)
                    
                    # Evaluate particle
                    particle_val = func(particle['position'])
                    evaluations += 1
                    
                    # Update personal best
                    if particle_val < particle['best_val']:
                        particle['best_val'] = particle_val
                        particle['best_pos'] = particle['position'].copy()
                    
                    # Update global best
                    if particle_val < self.global_best_val:
                        self.global_best_val = particle_val
                        self.global_best_pos = particle['position'].copy()
            
            # Dynamically adjust inertia weight
            self.inertia_weight = 0.9 - 0.5 * (evaluations / self.budget)
        
        return self.global_best_pos, self.global_best_val

    def initialize_swarm(self):
        swarm = []
        for _ in range(self.swarm_size):
            position = np.random.uniform(-5.0, 5.0, self.dim)
            velocity = np.random.uniform(-1.0, 1.0, self.dim)
            best_pos = position.copy()
            best_val = np.inf
            swarm.append({'position': position, 'velocity': velocity, 'best_pos': best_pos, 'best_val': best_val})
        return swarm