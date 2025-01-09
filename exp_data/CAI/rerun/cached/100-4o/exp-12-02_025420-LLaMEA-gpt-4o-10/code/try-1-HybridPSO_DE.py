import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim, swarm_size=30, de_cr=0.9, de_f=0.8):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.de_cr = de_cr
        self.de_f = de_f
        self.global_best_pos = None
        self.global_best_val = float('inf')
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Initialize the swarm
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_pos = np.copy(particles)
        personal_best_val = np.full(self.swarm_size, float('inf'))
        
        eval_count = 0
        while eval_count < self.budget:
            for i in range(self.swarm_size):
                if eval_count >= self.budget:
                    break
                # Evaluate the fitness of the current particle
                fitness = func(particles[i])
                eval_count += 1

                # Update personal best if the current fitness is better
                if fitness < personal_best_val[i]:
                    personal_best_val[i] = fitness
                    personal_best_pos[i] = particles[i]

                # Update global best if the current fitness is better
                if fitness < self.global_best_val:
                    self.global_best_val = fitness
                    self.global_best_pos = particles[i]

            # Update velocities and positions for the swarm
            w = 0.5 + 0.5 * (self.budget - eval_count) / self.budget  # Adaptive inertia weight
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = w * velocities[i] + 1.5 * r1 * (personal_best_pos[i] - particles[i]) + 1.5 * r2 * (self.global_best_pos - particles[i])
                particles[i] = particles[i] + velocities[i]
                
                # Ensure particles remain within bounds
                particles[i] = np.clip(particles[i], self.lower_bound, self.upper_bound)

            # Apply DE mutation and crossover
            for i in range(self.swarm_size):
                if eval_count >= self.budget:
                    break
                idxs = [idx for idx in range(self.swarm_size) if idx != i]
                a, b, c = particles[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.de_f * (b - c), self.lower_bound, self.upper_bound)
                cross_points = np.random.rand(self.dim) < self.de_cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, particles[i])
                
                # Evaluate the trial
                trial_fitness = func(trial)
                eval_count += 1

                # Perform selection
                if trial_fitness < personal_best_val[i]:
                    personal_best_val[i] = trial_fitness
                    personal_best_pos[i] = trial
                    particles[i] = trial

                # Update global best if needed
                if trial_fitness < self.global_best_val:
                    self.global_best_val = trial_fitness
                    self.global_best_pos = trial

        return self.global_best_pos, self.global_best_val