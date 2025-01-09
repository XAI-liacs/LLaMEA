import numpy as np

class EnhancedOppositionalParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, n_particles=30, w_max=0.9, w_min=0.4, c1_max=2.5, c1_min=1, c2_max=2.5, c2_min=1):
        self.budget = budget
        self.dim = dim
        self.n_particles = n_particles
        self.w_max = w_max
        self.w_min = w_min
        self.c1_max = c1_max
        self.c1_min = c1_min
        self.c2_max = c2_max
        self.c2_min = c2_min
        self.f_opt = np.Inf
        self.x_opt = None
        self.bounds = (-5.0, 5.0)
        
    def __call__(self, func):
        lb, ub = self.bounds
        particles = np.random.uniform(lb, ub, (self.n_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.n_particles, self.dim))
        pbest_positions = particles.copy()
        pbest_scores = np.full(self.n_particles, np.Inf)
        gbest_position = None
        gbest_score = np.Inf
        
        evaluations = 0

        while evaluations < self.budget:
            w = self.w_max - (self.w_max - self.w_min) * (evaluations / self.budget)
            c1 = self.c1_max - (self.c1_max - self.c1_min) * (evaluations / self.budget)
            c2 = self.c2_min + (self.c2_max - self.c2_min) * (evaluations / self.budget)

            for i in range(self.n_particles):
                score = func(particles[i])
                evaluations += 1
                if score < pbest_scores[i]:
                    pbest_scores[i] = score
                    pbest_positions[i] = particles[i]
                if score < gbest_score:
                    gbest_score = score
                    gbest_position = particles[i]

            if evaluations >= self.budget:
                break
            
            for i in range(self.n_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = c1 * r1 * (pbest_positions[i] - particles[i])
                social_component = c2 * r2 * (gbest_position - particles[i])
                velocities[i] = w * velocities[i] + cognitive_component + social_component
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], lb, ub)
                
                opposite_particles = lb + ub - particles[i]
                opposite_score = func(opposite_particles)
                evaluations += 1
                
                if opposite_score < score:
                    particles[i] = opposite_particles
                    if opposite_score < pbest_scores[i]:
                        pbest_scores[i] = opposite_score
                        pbest_positions[i] = particles[i]
                    if opposite_score < gbest_score:
                        gbest_score = opposite_score
                        gbest_position = particles[i]
                    
                # Random reinitialization for diversity
                if np.random.rand() < 0.1:
                    particles[i] = np.random.uniform(lb, ub, self.dim)
                    velocities[i] = np.random.uniform(-1, 1, self.dim)
                
                if evaluations >= self.budget:
                    break

        self.f_opt, self.x_opt = gbest_score, gbest_position
        return self.f_opt, self.x_opt