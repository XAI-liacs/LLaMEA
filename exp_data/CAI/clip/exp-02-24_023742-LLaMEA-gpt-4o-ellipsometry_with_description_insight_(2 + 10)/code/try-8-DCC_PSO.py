import numpy as np

class DCC_PSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evals = 0
    
    def __call__(self, func):
        # Initialize parameters
        num_particles = 10
        lb, ub = func.bounds.lb, func.bounds.ub
        w = 0.5  # inertia weight
        c1 = c2 = 2.0  # cognitive and social weight factors
        v_max = 0.2 * (ub - lb)  # maximum velocity
        
        # Initialize particles and velocities
        particles = np.random.uniform(lb, ub, (num_particles, self.dim))
        velocities = np.random.uniform(-v_max, v_max, (num_particles, self.dim))
        personal_best_positions = particles.copy()
        personal_best_scores = np.array([func(p) for p in particles])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        self.evals += num_particles

        # Run PSO
        while self.evals < self.budget:
            for i in range(num_particles):
                if self.evals >= self.budget:
                    break

                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (w * velocities[i] + 
                                 c1 * r1 * (personal_best_positions[i] - particles[i]) + 
                                 c2 * r2 * (global_best_position - particles[i]))
                
                # Constrict velocity
                velocities[i] = np.clip(velocities[i], -v_max, v_max)

                # Update particle position
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], lb, ub)

                # Evaluate fitness
                score = func(particles[i])
                self.evals += 1

                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i]

                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = particles[i]

        return global_best_position