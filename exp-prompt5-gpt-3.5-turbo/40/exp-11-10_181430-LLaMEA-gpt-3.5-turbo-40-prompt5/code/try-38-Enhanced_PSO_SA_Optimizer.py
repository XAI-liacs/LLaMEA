import numpy as np

class Enhanced_PSO_SA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 15
        self.max_iter = 1200
        self.c1 = 1.5
        self.c2 = 1.5
        self.inertia_weight = 0.6
        self.temp = 15.0
        self.alpha = 0.9
        self.diversity_threshold = 0.5
        self.pop_increase_factor = 1.2
        self.pop_decrease_factor = 0.8

    def __call__(self, func):
        def objective_function(x):
            return func(x)

        def accept_move(cost_diff, temp):
            return cost_diff < 0 or np.random.uniform(0, 1) < np.exp(-cost_diff / temp)

        particles = np.random.uniform(-5.0, 5.0, size=(self.pop_size, self.dim))
        velocities = np.zeros((self.pop_size, self.dim))
        personal_best = particles.copy()
        global_best = particles[np.argmin([objective_function(p) for p in particles])

        for _ in range(self.max_iter):
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = self.inertia_weight * velocities[i] + self.c1 * r1 * (personal_best[i] - particles[i]) + self.c2 * r2 * (global_best - particles[i])
                particles[i] = np.clip(particles[i] + velocities[i], -5.0, 5.0)

                for _ in range(5):
                    new_particle = particles[i] + np.random.normal(0, self.temp, size=self.dim)
                    cost_diff = objective_function(new_particle) - objective_function(particles[i])
                    if accept_move(cost_diff, self.temp):
                        particles[i] = new_particle

                if objective_function(particles[i]) < objective_function(personal_best[i]):
                    personal_best[i] = particles[i].copy()

                if objective_function(particles[i]) < objective_function(global_best):
                    global_best = particles[i].copy()

            self.temp *= self.alpha

            # Update population size based on fitness diversity
            fitness_values = [objective_function(p) for p in particles]
            diversity = np.std(fitness_values) / (np.max(fitness_values) - np.min(fitness_values))
            if diversity < self.diversity_threshold:
                self.pop_size = min(int(self.pop_size * self.pop_increase_factor), 50)
            else:
                self.pop_size = max(int(self.pop_size * self.pop_decrease_factor), 5)

            particles = np.concatenate([particles, np.random.uniform(-5.0, 5.0, size=(self.pop_size - len(particles), self.dim))])

        return global_best