import numpy as np

class HybridPSOLocalSearch:
    def __init__(self, budget=10000, dim=10, num_particles=20, max_iter=100):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.f_opt = np.Inf
        self.x_opt = None

    def local_search(self, x, func):
        # Perform a local search around current solution x
        # Update x and return the improved solution
        return x

    def __call__(self, func):
        swarm = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.num_particles, self.dim))
        velocities = np.random.uniform(-0.1, 0.1, size=(self.num_particles, self.dim))
        
        for i in range(self.max_iter):
            for j in range(self.num_particles):
                particle = swarm[j]
                velocity = velocities[j]
                new_particle = particle + velocity
                new_particle = np.clip(new_particle, func.bounds.lb, func.bounds.ub)
                
                if np.random.rand() < 0.1:
                    new_particle = self.local_search(new_particle, func)
                
                f = func(new_particle)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = new_particle

                swarm[j] = new_particle
                velocities[j] = velocity  # Update velocity based on PSO equations
                
            if i % 10 == 0:  # Optionally perform periodic local search for intensification
                for j in range(self.num_particles):
                    swarm[j] = self.local_search(swarm[j], func)
                    
        return self.f_opt, self.x_opt