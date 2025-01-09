import numpy as np

class PSO_DE:
    def __init__(self, budget=10000, dim=10, swarm_size=50, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        
        # Initialize swarm particles
        particles = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.array([func(p) for p in particles])
        
        # Global best
        global_best_score = np.min(personal_best_scores)
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]

        evaluations = self.swarm_size
        inertia_weight = 0.9  # Initial inertia weight

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                # Differential Evolution step
                candidates = np.random.choice(self.swarm_size, 3, replace=False)
                a, b, c = particles[candidates]
                mutant_vector = a + self.F * (b - c)
                mutant_vector = np.clip(mutant_vector, lb, ub)

                trial_vector = np.copy(particles[i])
                for j in range(self.dim):
                    if np.random.rand() < self.CR:
                        trial_vector[j] = mutant_vector[j]

                trial_score = func(trial_vector)
                evaluations += 1

                if trial_score < personal_best_scores[i]:
                    personal_best_positions[i] = trial_vector
                    personal_best_scores[i] = trial_score
                
                # Update global best
                if personal_best_scores[i] < global_best_score:
                    global_best_score = personal_best_scores[i]
                    global_best_position = personal_best_positions[i]

                # Particle Swarm Optimization step
                velocities[i] = inertia_weight * velocities[i] + \
                                1.5 * np.random.rand(self.dim) * (personal_best_positions[i] - particles[i]) + \
                                1.5 * np.random.rand(self.dim) * (global_best_position - particles[i])
                particles[i] = particles[i] + velocities[i]
                particles[i] = np.clip(particles[i], lb, ub)

                # Evaluate new position
                new_score = func(particles[i])
                evaluations += 1
                
                if new_score < personal_best_scores[i]:
                    personal_best_positions[i] = particles[i]
                    personal_best_scores[i] = new_score

                if new_score < global_best_score:
                    global_best_score = new_score
                    global_best_position = particles[i]

            inertia_weight = max(0.4, inertia_weight * 0.99)  # Decay inertia weight

            if evaluations >= self.budget:
                break

        self.f_opt = global_best_score
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt