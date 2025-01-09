import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 + int(2 * np.sqrt(self.dim))
        self.w = 0.9  # Start with higher inertia weight
        self.c1 = 1.5
        self.c2 = 1.5
        self.f = 0.8
        self.cr = 0.8  # Start with lower crossover probability

    def __call__(self, func):
        np.random.seed(42)
        particles = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dim)
        )
        velocities = np.random.uniform(
            -0.1, 0.1, (self.population_size, self.dim)
        )
        personal_best_positions = np.copy(particles)
        personal_best_values = np.full(self.population_size, np.inf)
        global_best_position = None
        global_best_value = np.inf
        
        evaluations = 0

        while evaluations < self.budget:
            # Adjust inertia weight and crossover probability dynamically
            w_min, w_max = 0.4, 0.9
            self.w = w_max - (w_max - w_min) * (evaluations / self.budget)
            self.cr = 0.9 - self.w  # Increase crossover probability as inertia weight decreases
            
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                current_value = func(particles[i])
                evaluations += 1
                if current_value < personal_best_values[i]:
                    personal_best_values[i] = current_value
                    personal_best_positions[i] = particles[i]
                if current_value < global_best_value:
                    global_best_value = current_value
                    global_best_position = particles[i]

            r1, r2 = np.random.rand(2)
            velocities = (
                self.w * velocities
                + self.c1 * r1 * (personal_best_positions - particles)
                + self.c2 * r2 * (global_best_position - particles)
            )
            particles += velocities
            particles = np.clip(particles, self.lower_bound, self.upper_bound)

            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                idxs = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = particles[idxs]
                mutant = np.clip(a + self.f * (b - c), self.lower_bound, self.upper_bound)
                
                crossover = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover, mutant, particles[i])
                
                trial_value = func(trial)
                evaluations += 1
                if trial_value < personal_best_values[i]:
                    personal_best_values[i] = trial_value
                    personal_best_positions[i] = trial
                    particles[i] = trial
                    if trial_value < global_best_value:
                        global_best_value = trial_value
                        global_best_position = trial

        return global_best_position, global_best_value