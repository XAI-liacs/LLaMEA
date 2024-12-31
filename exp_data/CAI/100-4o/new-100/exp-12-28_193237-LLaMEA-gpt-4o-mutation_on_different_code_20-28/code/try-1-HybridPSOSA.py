import numpy as np

class HybridPSOSA:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.num_particles = 50
        self.inertia_weight = 0.9  # Start with higher inertia
        self.cognitive_weight = 1.5
        self.social_weight = 1.5
        self.temperature = 1.0
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_pos = np.copy(particles)
        personal_best_val = np.array([func(p) for p in particles])
        global_best_pos = personal_best_pos[np.argmin(personal_best_val)]
        global_best_val = np.min(personal_best_val)
        
        evaluations = self.num_particles
        
        while evaluations < self.budget:
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_weight * r1 * (personal_best_pos[i] - particles[i]) +
                                 self.social_weight * r2 * (global_best_pos - particles[i]))
                particles[i] = np.clip(particles[i] + velocities[i], lb, ub)
                
                f_val = func(particles[i])
                evaluations += 1
                
                if f_val < personal_best_val[i]:
                    personal_best_pos[i] = particles[i]
                    personal_best_val[i] = f_val
                
                if f_val < global_best_val:
                    global_best_pos = particles[i]
                    global_best_val = f_val
            
            # Adaptive inertia weight adjustment
            self.inertia_weight = 0.4 + (0.5 * (self.budget - evaluations) / self.budget)
            
            # Simulated Annealing-like perturbation with dynamic scale
            perturb_scale = (ub - lb) * (0.5 * (self.budget - evaluations) / self.budget)
            for i in range(self.num_particles):
                if np.random.rand() < np.exp(-abs(global_best_val - personal_best_val[i]) / self.temperature):
                    perturbation = global_best_pos + np.random.uniform(-perturb_scale, perturb_scale, self.dim)
                    perturbation = np.clip(perturbation, lb, ub)
                    perturbed_f_val = func(perturbation)
                    evaluations += 1
                    if perturbed_f_val < global_best_val:
                        global_best_pos = perturbation
                        global_best_val = perturbed_f_val
            
            self.temperature *= 0.99  # cooling schedule
            
            if evaluations >= self.budget:
                break
        
        self.f_opt = global_best_val
        self.x_opt = global_best_pos
        return self.f_opt, self.x_opt