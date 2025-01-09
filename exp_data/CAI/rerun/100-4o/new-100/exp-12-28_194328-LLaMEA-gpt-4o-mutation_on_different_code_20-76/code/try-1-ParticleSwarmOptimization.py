import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, swarm_size=30):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        velocity_clamp_base = (ub - lb) * 0.1
        
        # Initialize swarm
        swarm = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-abs(velocity_clamp_base), abs(velocity_clamp_base), (self.swarm_size, self.dim))
        personal_best_positions = np.copy(swarm)
        personal_best_scores = np.array([func(x) for x in swarm])
        
        # Initialize global best
        global_best_score = np.min(personal_best_scores)
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        
        # Main loop
        for t in range(self.budget // self.swarm_size):
            velocity_clamp = velocity_clamp_base * (0.9 ** (t / 100))
            for i in range(self.swarm_size):
                # Update velocity
                inertia = 0.5 + np.random.rand() / 2
                cognitive_component = np.random.rand(self.dim) * 1.5  # Increased personal learning factor
                social_component = np.random.rand(self.dim)
                velocities[i] = (
                    inertia * velocities[i]
                    + cognitive_component * (personal_best_positions[i] - swarm[i])
                    + social_component * (global_best_position - swarm[i])
                )
                # Clamp velocity
                velocities[i] = np.clip(velocities[i], -abs(velocity_clamp), abs(velocity_clamp))
                
                # Update position
                swarm[i] += velocities[i]
                swarm[i] = np.clip(swarm[i], lb, ub)
                
                # Evaluate fitness
                f = func(swarm[i])
                
                # Update personal best
                if f < personal_best_scores[i]:
                    personal_best_scores[i] = f
                    personal_best_positions[i] = swarm[i]
                
                # Update global best
                if f < global_best_score:
                    global_best_score = f
                    global_best_position = swarm[i]
        
        self.f_opt = global_best_score
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt