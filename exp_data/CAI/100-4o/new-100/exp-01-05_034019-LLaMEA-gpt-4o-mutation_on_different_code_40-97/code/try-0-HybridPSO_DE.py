import numpy as np

class HybridPSO_DE:
    def __init__(self, budget=10000, dim=10, pop_size=50):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.inertia = 0.5
        self.cognitive_component = 1.5
        self.social_component = 1.5
        self.f = 0.5  # Differential evolution scaling factor
        self.cr = 0.9  # Crossover probability

    def __call__(self, func):
        # Initialize particles and velocities
        particles = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best_positions = particles.copy()
        personal_best_values = np.full(self.pop_size, np.Inf)

        for _ in range(self.budget):
            # Evaluate current positions
            for i in range(self.pop_size):
                fitness = func(particles[i])
                if fitness < personal_best_values[i]:
                    personal_best_values[i] = fitness
                    personal_best_positions[i] = particles[i].copy()
                if fitness < self.f_opt:
                    self.f_opt = fitness
                    self.x_opt = particles[i].copy()

            global_best_position = personal_best_positions[np.argmin(personal_best_values)]

            # Update velocities and positions (Particle Swarm Optimization)
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.inertia * velocities[i] +
                                 self.cognitive_component * r1 * (personal_best_positions[i] - particles[i]) +
                                 self.social_component * r2 * (global_best_position - particles[i]))
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], func.bounds.lb, func.bounds.ub)

            # Adaptive Differential Evolution component
            for i in range(self.pop_size):
                idxs = [index for index in range(self.pop_size) if index != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = personal_best_positions[a] + self.f * (personal_best_positions[b] - personal_best_positions[c])
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
                cross_points = np.random.rand(self.dim) < self.cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, particles[i])
                trial_fitness = func(trial)
                if trial_fitness < personal_best_values[i]:
                    personal_best_values[i] = trial_fitness
                    personal_best_positions[i] = trial.copy()
                if trial_fitness < self.f_opt:
                    self.f_opt = trial_fitness
                    self.x_opt = trial.copy()

        return self.f_opt, self.x_opt