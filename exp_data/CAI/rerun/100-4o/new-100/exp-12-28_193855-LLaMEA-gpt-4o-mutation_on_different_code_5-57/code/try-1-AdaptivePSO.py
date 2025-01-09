import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, swarm_size=50, c1=2.0, c2=2.0):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.c1 = c1
        self.c2 = c2
        self.w = 0.9  # Start with a higher inertia weight
        self.w_min = 0.4
        self.w_max = 0.9
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize particles
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.zeros_like(particles)
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.swarm_size, np.inf)
        
        # Evaluate initial particles
        for i in range(self.swarm_size):
            score = func(particles[i])
            if score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = particles[i]
            if score < self.f_opt:
                self.f_opt = score
                self.x_opt = particles[i]

        evaluations = self.swarm_size
        success_rate = 0.0
        
        # Begin optimization loop
        while evaluations < self.budget:
            # Update inertia weight adaptively with stochastic component
            stochastic_noise = np.random.uniform(-0.1, 0.1)
            self.w = self.w_max - (self.w_max - self.w_min) * success_rate + stochastic_noise

            for i in range(self.swarm_size):
                # Update velocity and position
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - particles[i]) +
                                 self.c2 * r2 * (self.x_opt - particles[i]))
                
                particles[i] = np.clip(particles[i] + velocities[i], lb, ub)
                
                # Evaluate particle
                score = func(particles[i])
                evaluations += 1
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i]
                    if score < self.f_opt:
                        self.f_opt = score
                        self.x_opt = particles[i]
                        success_rate += 1 / self.swarm_size

                # Check budget
                if evaluations >= self.budget:
                    break

            # Decay success rate gradually
            success_rate *= 0.95

        return self.f_opt, self.x_opt