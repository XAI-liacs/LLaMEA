import numpy as np

class HybridPSOSA:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 50
        self.initial_inertia_weight = 0.9  # Change 1
        self.final_inertia_weight = 0.4  # Change 2
        self.cognitive_const = 1.5
        self.social_const = 1.5
        self.temperature = 100
        self.cooling_rate = 0.99
        self.bounds_lb = -5.0
        self.bounds_ub = 5.0
    
    def __call__(self, func):
        particles = np.random.uniform(self.bounds_lb, self.bounds_ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.population_size, np.inf)
        
        global_best_score = np.inf
        global_best_position = None
        
        eval_count = 0
        iteration = 0  # New variable for iterations

        while eval_count < self.budget:
            inertia_weight = self.initial_inertia_weight - (self.initial_inertia_weight - self.final_inertia_weight) * (eval_count / self.budget)  # Change 3

            for i, particle in enumerate(particles):
                score = func(particle)
                eval_count += 1
                
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particle
                
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = particle
            
            for i in range(self.population_size):
                inertia = inertia_weight * velocities[i]
                cognitive = self.cognitive_const * np.random.rand(self.dim) * (personal_best_positions[i] - particles[i])
                social = self.social_const * np.random.rand(self.dim) * (global_best_position - particles[i])
                
                velocities[i] = inertia + cognitive + social
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], self.bounds_lb, self.bounds_ub)
            
            for i in range(self.population_size):
                candidate_position = particles[i] + np.random.uniform(-0.5, 0.5, self.dim)
                candidate_position = np.clip(candidate_position, self.bounds_lb, self.bounds_ub)
                candidate_score = func(candidate_position)
                eval_count += 1
                
                if candidate_score < personal_best_scores[i] or np.random.rand() < np.exp((personal_best_scores[i] - candidate_score) / self.temperature):
                    personal_best_scores[i] = candidate_score
                    personal_best_positions[i] = candidate_position

            if iteration % (self.budget // (2 * self.population_size)) == 0 and eval_count < self.budget * 0.9:  # Change 4
                random_restart = np.random.uniform(self.bounds_lb, self.bounds_ub, (self.population_size, self.dim))  # Change 5
                particles = np.where(personal_best_scores.reshape(-1, 1) > global_best_score, random_restart, particles)  # Change 6

            self.temperature *= self.cooling_rate
            
            if np.min(personal_best_scores) < global_best_score:
                global_best_score = np.min(personal_best_scores)
                global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
            
            iteration += 1  # Increment iteration count
        
        self.f_opt = global_best_score
        self.x_opt = global_best_position
        
        return self.f_opt, self.x_opt