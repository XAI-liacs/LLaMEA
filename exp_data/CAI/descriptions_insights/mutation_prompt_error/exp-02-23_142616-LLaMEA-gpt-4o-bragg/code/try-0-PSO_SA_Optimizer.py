import numpy as np

class PSO_SA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = min(50, budget // 10)
        self.w = 0.5  # inertia weight
        self.c1 = 1.5  # cognitive (particle) weight
        self.c2 = 1.5  # social (swarm) weight
        self.temperature_initial = 1000
        self.temperature_final = 1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = particles.copy()
        personal_best_scores = np.array([func(p) for p in particles])
        global_best_position = personal_best_positions[personal_best_scores.argmin()]
        
        evaluations = self.num_particles
        global_best_score = personal_best_scores.min()
        
        while evaluations < self.budget:
            temperature = self.temperature_initial * ((self.temperature_final / self.temperature_initial) ** (evaluations / self.budget))
            
            for i, particle in enumerate(particles):
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * np.random.rand(self.dim) * (personal_best_positions[i] - particle) +
                                 self.c2 * np.random.rand(self.dim) * (global_best_position - particle))
                particles[i] = np.clip(particles[i] + velocities[i], lb, ub)
                
                current_score = func(particles[i])
                evaluations += 1
                
                if current_score < personal_best_scores[i]:
                    personal_best_positions[i] = particles[i].copy()
                    personal_best_scores[i] = current_score
                    
                if current_score < global_best_score:
                    global_best_position = particles[i].copy()
                    global_best_score = current_score
                else:
                    acceptance_prob = np.exp((global_best_score - current_score) / temperature)
                    if np.random.rand() < acceptance_prob:
                        global_best_position = particles[i].copy()
                        global_best_score = current_score
                    
                if evaluations >= self.budget:
                    break
        
        return global_best_position, global_best_score