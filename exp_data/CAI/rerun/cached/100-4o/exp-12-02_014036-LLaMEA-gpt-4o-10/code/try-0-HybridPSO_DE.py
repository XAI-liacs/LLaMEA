import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = 20  # population size
        self.inertia = 0.7
        self.c1 = 1.5
        self.c2 = 1.5
        self.F = 0.8  # differential evolution parameter
        self.CR = 0.9  # crossover probability
        
    def __call__(self, func):
        # Initialize population of particles and velocities
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.pop_size, float('inf'))
        
        # Evaluate initial population
        scores = np.apply_along_axis(func, 1, particles)
        self.budget -= self.pop_size
        
        # Update personal bests
        better_mask = scores < personal_best_scores
        personal_best_scores[better_mask] = scores[better_mask]
        personal_best_positions[better_mask] = particles[better_mask]
        
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]
        
        while self.budget > 0:
            # Particle Swarm Optimization step
            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)
            velocities = (self.inertia * velocities +
                          self.c1 * r1 * (personal_best_positions - particles) +
                          self.c2 * r2 * (global_best_position - particles))
            particles = np.clip(particles + velocities, self.lower_bound, self.upper_bound)
            
            # Differential Evolution step
            for i in range(self.pop_size):
                if self.budget <= 0:
                    break
                # Mutate
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = particles[a] + self.F * (particles[b] - particles[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, particles[i])
                
                # Selection
                trial_score = func(trial)
                self.budget -= 1
                if trial_score < scores[i]:
                    scores[i] = trial_score
                    particles[i] = trial
                    if trial_score < personal_best_scores[i]:
                        personal_best_scores[i] = trial_score
                        personal_best_positions[i] = trial
                        if trial_score < personal_best_scores[global_best_index]:
                            global_best_index = i
                            global_best_position = trial
        
        return global_best_position, personal_best_scores[global_best_index]