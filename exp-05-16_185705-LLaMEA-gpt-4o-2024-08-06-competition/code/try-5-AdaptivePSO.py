import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, swarm_size=30):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size

    def __call__(self, func):
        # Initialize particles
        particles = np.random.uniform(-100, 100, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        
        # Initialize personal best
        personal_best = particles.copy()
        personal_best_values = np.array([func(p) for p in personal_best])
        
        # Initialize global best
        global_best_idx = np.argmin(personal_best_values)
        global_best = personal_best[global_best_idx]
        global_best_value = personal_best_values[global_best_idx]
        
        evaluations = self.swarm_size
        
        while evaluations < self.budget:
            # Adaptive parameters
            diversity = np.mean(np.std(particles, axis=0))
            inertia_weight = max(0.4, 0.9 * (1 - diversity / 50.0))
            cognitive_component = np.random.uniform(1.4, np.min([2.4, 2.0 + diversity / 25.0]))  # Changed line
            social_component = np.random.uniform(1.4, np.min([2.4, 2.0 + diversity / 25.0]))  # Changed line
            
            # Update velocities and particles
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(2)
                velocities[i] = (inertia_weight * velocities[i] 
                                 + cognitive_component * r1 * (personal_best[i] - particles[i])
                                 + social_component * r2 * (global_best - particles[i]))
                
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], -100, 100)
                
                # Evaluate new position
                current_value = func(particles[i])
                evaluations += 1
                
                # Update personal best
                if current_value < personal_best_values[i]:
                    personal_best[i] = particles[i]
                    personal_best_values[i] = current_value
                    
                    # Update global best
                    if current_value < global_best_value:
                        global_best_value = current_value
                        global_best = particles[i]
                        
                if evaluations >= self.budget:
                    break
        
        return global_best_value, global_best