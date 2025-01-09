import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, num_particles=30):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize particles' positions and velocities
        pos = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.num_particles, self.dim))
        vel = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_pos = pos.copy()
        personal_best_value = np.full(self.num_particles, np.inf)
        
        # Initialize global best
        global_best_pos = None
        global_best_value = np.inf
        
        # PSO parameters
        w_initial = 0.9
        w_final = 0.4
        c1, c2 = 2.0, 2.0
        
        evals = 0
        
        while evals < self.budget:
            inertia_weight = w_initial - (w_initial - w_final) * (evals / self.budget)
            
            for i in range(self.num_particles):
                if evals >= self.budget:
                    break
                
                # Evaluate the fitness of each particle
                f_value = func(pos[i])
                evals += 1
                
                # Update the personal best
                if f_value < personal_best_value[i]:
                    personal_best_value[i] = f_value
                    personal_best_pos[i] = pos[i].copy()
                
                # Update the global best
                if f_value < global_best_value:
                    global_best_value = f_value
                    global_best_pos = pos[i].copy()
            
            # Update velocities and positions
            for i in range(self.num_particles):
                if evals >= self.budget:
                    break
                
                r1, r2 = np.random.uniform(0, 1, 2)
                vel[i] = (inertia_weight * vel[i] +
                          c1 * r1 * (personal_best_pos[i] - pos[i]) +
                          c2 * r2 * (global_best_pos - pos[i]))
                
                # Clamp velocity to keep within bounds
                vel[i] = np.clip(vel[i], -2, 2)
                
                # Update position
                pos[i] = pos[i] + vel[i]
                
                # Ensure position is within the search space
                pos[i] = np.clip(pos[i], func.bounds.lb, func.bounds.ub)
        
        self.f_opt = global_best_value
        self.x_opt = global_best_pos
        return self.f_opt, self.x_opt