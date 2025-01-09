import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)
        self.pop_size = 20
        self.max_iter = budget // self.pop_size
        self.c1 = 1.5  # Cognitive component
        self.c2 = 1.5  # Social component
        self.w = 0.5   # Inertia weight
        self.F = 0.8   # DE scaling factor
        self.CR = 0.9  # DE crossover probability

    def __call__(self, func):
        # Initialize particles
        particles = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best_positions = particles.copy()
        personal_best_scores = np.full(self.pop_size, np.inf)

        # Initial evaluation
        for i in range(self.pop_size):
            score = func(particles[i])
            personal_best_scores[i] = score

        # Get global best
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index].copy()
        
        for iter in range(self.max_iter):
            for i in range(self.pop_size):
                # Update velocity and position (PSO step)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * np.random.rand() * (personal_best_positions[i] - particles[i]) +
                                 self.c2 * np.random.rand() * (global_best_position - particles[i]))
                particles[i] = particles[i] + velocities[i]
                particles[i] = np.clip(particles[i], self.bounds[0], self.bounds[1])

                # Mutate and crossover (DE step)
                idxs = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = particles[idxs]
                mutant = np.clip(a + self.F * (b - c), self.bounds[0], self.bounds[1])
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, particles[i])

                # Evaluate trial
                trial_score = func(trial)
                if trial_score < personal_best_scores[i]:
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial

                # Update global best
                if trial_score < func(global_best_position):
                    global_best_position = trial

        return global_best_position

# Example of usage:
# optimizer = HybridPSO_DE(budget=1000, dim=10)
# best_solution = optimizer(some_black_box_function)