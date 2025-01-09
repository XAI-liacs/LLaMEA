import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, num_particles=30, inertia=0.7, cognitive=1.5, social=1.5):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = particles.copy()
        personal_best_scores = np.full(self.num_particles, np.Inf)
        
        global_best_position = None
        global_best_score = np.Inf

        eval_count = 0

        while eval_count < self.budget:
            for i in range(self.num_particles):
                if eval_count >= self.budget:
                    break
                
                score = func(particles[i])
                eval_count += 1
                
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i]

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = particles[i]

            r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
            self.inertia = 0.9 - 0.5 * (eval_count / self.budget)  # Adaptive inertia
            velocities = (self.inertia * velocities
                          + self.cognitive * r1 * (personal_best_positions - particles)
                          + self.social * r2 * (global_best_position - particles))
            
            particles = particles + velocities
            particles = np.clip(particles, lb, ub)

        self.f_opt = global_best_score
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt