import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 40
        self.w = 0.5  # Inertia weight
        self.c1 = 1.5 # Cognitive coefficient
        self.c2 = 1.5 # Social coefficient
        self.CR = 0.9 # Crossover rate for DE
        self.F = 0.8  # Differential weight for DE
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        np.random.seed(42)
        num_evaluations = 0

        # Initialize the swarm
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.array([func(p) for p in particles])
        num_evaluations += self.population_size
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]

        while num_evaluations < self.budget:
            # PSO Step
            r1, r2 = np.random.rand(2)
            velocities = self.w * velocities + \
                         self.c1 * r1 * (personal_best_positions - particles) + \
                         self.c2 * r2 * (global_best_position - particles)
            particles = np.clip(particles + velocities, self.lower_bound, self.upper_bound)

            # Evaluate the particles
            scores = np.array([func(p) for p in particles])
            num_evaluations += self.population_size

            # Update personal bests
            better_mask = scores < personal_best_scores
            personal_best_positions[better_mask] = particles[better_mask]
            personal_best_scores[better_mask] = scores[better_mask]

            # Update global best
            current_best_idx = np.argmin(personal_best_scores)
            current_best_score = personal_best_scores[current_best_idx]
            if current_best_score < global_best_score:
                global_best_score = current_best_score
                global_best_position = personal_best_positions[current_best_idx]

            # DE Step
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = particles[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, particles[i])
                f_trial = func(trial)
                num_evaluations += 1
                if f_trial < scores[i]:
                    particles[i] = trial
                    scores[i] = f_trial
                    if f_trial < global_best_score:
                        global_best_score = f_trial
                        global_best_position = trial

            if num_evaluations >= self.budget:
                break
        
        return global_best_position, global_best_score