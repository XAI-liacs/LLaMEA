import numpy as np

class HybridPSO_DE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 50
        self.w = 0.5  # inertia weight for PSO
        self.c1 = 1.5  # cognitive component
        self.c2 = 1.5  # social component
        self.F = 0.8  # mutation factor for DE
        self.CR = 0.9  # crossover probability for DE

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Initialize particle positions and velocities
        particles = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.population_size, np.inf)

        # Evaluate initial population
        scores = np.array([func(p) for p in particles])
        self.budget -= self.population_size

        # Update personal bests
        for i in range(self.population_size):
            if scores[i] < personal_best_scores[i]:
                personal_best_scores[i] = scores[i]
                personal_best_positions[i] = particles[i]

        # Determine global best
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        self.f_opt = personal_best_scores[global_best_idx]
        self.x_opt = global_best_position

        while self.budget > 0:
            # Particle Swarm Optimization step
            r1, r2 = np.random.rand(2, self.population_size, self.dim)
            velocities = (self.w * velocities 
                          + self.c1 * r1 * (personal_best_positions - particles) 
                          + self.c2 * r2 * (global_best_position - particles))
            particles = np.clip(particles + velocities, lb, ub)

            # Differential Evolution step
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = particles[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), lb, ub)
                trial = np.copy(particles[i])
                jrand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == jrand:
                        trial[j] = mutant[j]

                # Evaluate trial particle
                f_trial = func(trial)
                self.budget -= 1
                if f_trial < scores[i]:
                    particles[i] = trial
                    scores[i] = f_trial

                    # Update personal best if necessary
                    if f_trial < personal_best_scores[i]:
                        personal_best_scores[i] = f_trial
                        personal_best_positions[i] = trial

                        # Update global best if necessary
                        if f_trial < self.f_opt:
                            self.f_opt = f_trial
                            self.x_opt = trial

                if self.budget <= 0:
                    break

        return self.f_opt, self.x_opt