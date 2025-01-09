import numpy as np

class PSO_SA:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None
        self.c1 = 1.5  # cognitive component
        self.c2 = 1.5  # social component
        self.w_start = 0.9  # initial inertia weight
        self.w_end = 0.4  # final inertia weight
        self.temp_start = 1.0  # initial temperature for simulated annealing
        self.temp_end = 0.1  # final temperature
        self.cooling_rate = 0.99  # initial cooling rate for simulated annealing

    def __call__(self, func):
        num_particles = 30
        lb, ub = func.bounds.lb, func.bounds.ub
        
        # Initialize particles
        particles = np.random.uniform(lb, ub, (num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (num_particles, self.dim))
        
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(num_particles, np.inf)
        
        global_best_position = None
        global_best_score = np.inf

        evaluations = 0
        
        while evaluations < self.budget:
            # Adaptive inertia weight and temperature
            self.w = self.w_start - (self.w_start - self.w_end) * (evaluations / self.budget)
            self.temp = self.temp_start - (self.temp_start - self.temp_end) * (evaluations / self.budget)
            
            for i in range(num_particles):
                if evaluations >= self.budget:
                    break
                
                # Evaluate particle
                score = func(particles[i])
                evaluations += 1
                
                # Update personal bests
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i]
                
                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = particles[i]
            
            # Update velocities and positions
            for i in range(num_particles):
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.w * velocities[i] +
                                self.c1 * r1 * (personal_best_positions[i] - particles[i]) +
                                self.c2 * r2 * (global_best_position - particles[i]))
                
                # Simulated annealing inspired update
                new_position = particles[i] + velocities[i]
                new_position = np.clip(new_position, lb, ub)
                new_score = func(new_position)
                evaluations += 1
                
                if new_score < score or np.random.rand() < np.exp((score - new_score) / self.temp):
                    particles[i] = new_position
                    score = new_score
                
                if score < self.f_opt:
                    self.f_opt = score
                    self.x_opt = particles[i]
        
        return self.f_opt, self.x_opt