import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        
        # Parameters
        num_particles = 30
        w_initial, w_final = 0.9, 0.4  # Adaptive inertia weight
        c1_initial, c2_initial = 1.5, 1.5  # Adaptive cognitive and social constants
        F = 0.85  # DE scaling factor
        CR = 0.9  # DE crossover probability
        
        # Initialize particles
        particles = np.random.uniform(lb, ub, (num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (num_particles, self.dim))
        personal_best = particles.copy()
        personal_best_value = np.array([func(p) for p in particles])
        global_best = personal_best[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)
        
        evaluations = num_particles
        
        while evaluations < self.budget:
            for i in range(num_particles):
                # Update velocity and position using PSO
                w = w_final + (w_initial - w_final) * (1 - evaluations / self.budget)
                c1 = c1_initial + (2.0 - c1_initial) * (evaluations / self.budget)
                c2 = c2_initial + (2.0 - c2_initial) * (evaluations / self.budget)

                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (w * velocities[i] +
                                 c1 * r1 * (personal_best[i] - particles[i]) +
                                 c2 * r2 * (global_best - particles[i]))
                particles[i] = np.clip(particles[i] + velocities[i], lb, ub)
                
                # Evaluate new position
                particle_value = func(particles[i])
                evaluations += 1
                
                # Update personal best
                if particle_value < personal_best_value[i]:
                    personal_best[i] = particles[i]
                    personal_best_value[i] = particle_value
                    
                    # Update global best
                    if particle_value < global_best_value:
                        global_best = particles[i]
                        global_best_value = particle_value

                # Apply Differential Evolution strategy
                if evaluations < self.budget:
                    indices = list(range(num_particles))
                    indices.remove(i)
                    a, b, c = np.random.choice(indices, 3, replace=False)
                    if np.linalg.norm(personal_best[a] - personal_best[i]) < np.linalg.norm(personal_best[b] - personal_best[c]):
                        mutant_vector = personal_best[a] + F * (personal_best[b] - personal_best[c])
                    else:
                        mutant_vector = personal_best[c] + F * (personal_best[b] - personal_best[a])
                        
                    trial_vector = np.where(np.random.rand(self.dim) < CR, mutant_vector, particles[i])
                    trial_vector = np.clip(trial_vector, lb, ub)
                    
                    trial_value = func(trial_vector)
                    evaluations += 1

                    # Update particle with trial vector if it's better
                    if trial_value < particle_value:
                        particles[i] = trial_vector
                        personal_best[i] = trial_vector
                        personal_best_value[i] = trial_value
                        
                        # Update global best
                        if trial_value < global_best_value:
                            global_best = trial_vector
                            global_best_value = trial_value

        return global_best