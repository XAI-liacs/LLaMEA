import numpy as np

class PSO_DE_Optimizer:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.num_particles = 50
        self.c1 = 1.5  # cognitive coefficient
        self.c2 = 1.5  # social coefficient
        self.w = 0.7  # inertia weight
        self.F = 0.6  # differential weight
        self.CR = 0.9  # crossover probability

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        particles = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        velocities = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_values = np.array([func(p) for p in particles])
        global_best_position = personal_best_positions[np.argmin(personal_best_values)]
        global_best_value = np.min(personal_best_values)
        
        evaluations = self.num_particles

        while evaluations < self.budget:
            for i in range(self.num_particles):
                # PSO update
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.w * velocities[i] + 
                                 self.c1 * r1 * (personal_best_positions[i] - particles[i]) + 
                                 self.c2 * r2 * (global_best_position - particles[i]))
                particles[i] += velocities[i]

                # Ensure particles are within bounds
                particles[i] = np.clip(particles[i], lb, ub)

                # Evaluate current particle
                current_value = func(particles[i])
                evaluations += 1

                # Update personal best
                if current_value < personal_best_values[i]:
                    personal_best_values[i] = current_value
                    personal_best_positions[i] = particles[i]

                # Update global best
                if current_value < global_best_value:
                    global_best_value = current_value
                    global_best_position = particles[i]

            # DE update
            for i in range(self.num_particles):
                idxs = [idx for idx in range(self.num_particles) if idx != i]
                a, b, c = particles[np.random.choice(idxs, 3, replace=False)]
                mutant_vector = np.clip(a + self.F * (b - c), lb, ub)
                trial_vector = np.where(np.random.rand(self.dim) < self.CR, mutant_vector, particles[i])
                
                trial_value = func(trial_vector)
                evaluations += 1
                
                if trial_value < personal_best_values[i]:
                    personal_best_values[i] = trial_value
                    personal_best_positions[i] = trial_vector

                    if trial_value < global_best_value:
                        global_best_value = trial_value
                        global_best_position = trial_vector

            if evaluations >= self.budget:
                break

        self.f_opt = global_best_value
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt