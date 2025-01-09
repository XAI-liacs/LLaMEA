import numpy as np

class EnhancedParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, num_particles=30, inertia_weight=0.7, cognitive_weight=1.5, social_weight=1.5, inertia_decay=0.99):
        self.budget = budget
        self.dim = dim
        self.num_particles = num_particles
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.inertia_decay = inertia_decay
        
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        particles = np.random.uniform(bounds[0], bounds[1], size=(self.num_particles, self.dim))
        velocities = np.zeros((self.num_particles, self.dim))
        personal_best = particles.copy()
        personal_best_scores = np.array([func(p) for p in personal_best])
        global_best = personal_best[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)
        
        for _ in range(self.budget):
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = self.inertia_weight * velocities[i] + \
                                self.cognitive_weight * r1 * (personal_best[i] - particles[i]) + \
                                self.social_weight * r2 * (global_best - particles[i])
                particles[i] = np.clip(particles[i] + velocities[i], bounds[0], bounds[1])
                scores = np.array([func(p) for p in particles])
                
                for j, score in enumerate(scores):
                    if score < personal_best_scores[j]:
                        personal_best[j] = particles[j]
                        personal_best_scores[j] = score
                        
                best_idx = np.argmin(personal_best_scores)
                if personal_best_scores[best_idx] < global_best_score:
                    global_best = personal_best[best_idx]
                    global_best_score = personal_best_scores[best_idx]
            
            self.inertia_weight *= self.inertia_decay  # Dynamic inertia weight update
        
        self.f_opt = global_best_score
        self.x_opt = global_best
        
        return self.f_opt, self.x_opt