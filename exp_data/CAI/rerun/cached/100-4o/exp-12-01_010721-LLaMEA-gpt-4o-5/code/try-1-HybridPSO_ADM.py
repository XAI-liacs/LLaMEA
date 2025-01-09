import numpy as np

class HybridPSO_ADM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.num_particles = 10 + int(np.sqrt(self.dim))
        self.w = 0.5
        self.c1 = 1.5
        self.c2 = 1.5
        self.mutation_factor = 0.8
        self.crossover_prob = 0.9
    
    def __call__(self, func):
        np.random.seed(42)
        particles = np.random.uniform(self.lb, self.ub, (self.num_particles, self.dim))
        velocities = np.zeros((self.num_particles, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.num_particles, np.inf)
        
        global_best_position = None
        global_best_score = np.inf
        
        eval_count = 0
        
        while eval_count < self.budget:
            # Evaluate current particles
            for i in range(self.num_particles):
                score = func(particles[i])
                eval_count += 1
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i]
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = particles[i]

            # Update velocities and positions
            r1, r2 = np.random.random((2, self.num_particles, self.dim))
            velocities = (self.w * velocities +
                          self.c1 * r1 * (personal_best_positions - particles) +
                          self.c2 * r2 * (global_best_position - particles))
            particles += velocities
            
            # Apply bounds
            particles = np.clip(particles, self.lb, self.ub)
            
            # Adaptive Differential Mutation
            for i in range(self.num_particles):
                if eval_count >= self.budget:
                    break
                indices = np.random.choice(self.num_particles, 3, replace=False)
                if i in indices:
                    continue
                a, b, c = personal_best_positions[indices]
                mutant = np.clip(a + self.mutation_factor * (b - c), self.lb, self.ub)
                trial = np.where(np.random.rand(self.dim) < self.crossover_prob, mutant, particles[i])
                trial_score = func(trial)
                eval_count += 1
                if trial_score < personal_best_scores[i]:
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial
                if trial_score < global_best_score:
                    global_best_score = trial_score
                    global_best_position = trial
                
        return global_best_position, global_best_score