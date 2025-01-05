import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, population_size=30, inertia_weight=0.7, cognitive_coeff=1.5, social_coeff=1.5):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.inertia_weight = inertia_weight
        self.cognitive_coeff = cognitive_coeff
        self.social_coeff = social_coeff
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.population_size, np.Inf)
        global_best_position = None
        global_best_score = np.Inf

        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                score = func(particles[i])
                evaluations += 1
                
                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i]
                
                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = particles[i]
            
            # Update velocities and positions
            r1, r2 = np.random.rand(2)
            velocities = (self.inertia_weight * velocities +
                          self.cognitive_coeff * r1 * (personal_best_positions - particles) +
                          self.social_coeff * r2 * (global_best_position - particles))
            particles = particles + velocities
            particles = np.clip(particles, lb, ub)

        self.f_opt = global_best_score
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt