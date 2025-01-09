import numpy as np

class ParticleAccelerationOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lower_bound = -5.0
        upper_bound = 5.0
        swarm_size = 30
        max_iters = self.budget // swarm_size

        # Initialize particle positions and velocities
        particles = np.random.uniform(low=lower_bound, high=upper_bound, size=(swarm_size, self.dim))
        velocities = np.zeros((swarm_size, self.dim))
        best_particle = particles[np.argmin([func(p) for p in particles])]
        best_value = func(best_particle)

        for _ in range(max_iters):
            for i in range(swarm_size):
                acceleration = np.random.uniform() * (best_particle - particles[i]) + np.random.uniform() * (particles[np.argmin([func(p) for p in particles])] - particles[i])
                velocities[i] += acceleration
                particles[i] += velocities[i]

                # Update best particle and value
                current_value = func(particles[i])
                if current_value < best_value:
                    best_particle = particles[i]
                    best_value = current_value

        return best_particle