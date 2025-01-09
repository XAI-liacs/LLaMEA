import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = 30
        self.de_cross_prob = 0.9
        self.gbest_position = None
        self.gbest_value = float('inf')

    def initialize_population(self):
        self.population = self.lb + (self.ub - self.lb) * np.random.rand(self.population_size, self.dim)
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.pbest_positions = np.copy(self.population)
        self.pbest_values = np.array([float('inf')] * self.population_size)

    def evaluate_population(self, func):
        for i in range(self.population_size):
            value = func(self.population[i])
            if value < self.pbest_values[i]:
                self.pbest_values[i] = value
                self.pbest_positions[i] = self.population[i]
            if value < self.gbest_value:
                self.gbest_value = value
                self.gbest_position = self.population[i]

    def update_particles(self):
        w_max = 0.9  # maximum inertia weight
        w_min = 0.4  # minimum inertia weight
        w = w_max - ((w_max - w_min) * (self.gbest_value / self.budget))  # adaptive inertia weight
        c1 = 1.5  # cognitive component
        c2 = 1.5  # social component
        for i in range(self.population_size):
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)
            cognitive_velocity = c1 * r1 * (self.pbest_positions[i] - self.population[i])
            social_velocity = c2 * r2 * (self.gbest_position - self.population[i])
            self.velocities[i] = w * self.velocities[i] + cognitive_velocity + social_velocity
            self.population[i] = self.population[i] + self.velocities[i]
            self.population[i] = np.clip(self.population[i], self.lb, self.ub)
            if np.random.rand() < 0.1:  # random perturbation
                self.population[i] += np.random.uniform(-0.1, 0.1, self.dim)

    def differential_evolution_crossover(self):
        for i in range(self.population_size):
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
            mutant_vector = np.clip(a + 0.8 * (b - c), self.lb, self.ub)
            rand_idx = np.random.randint(self.dim)
            trial_vector = np.array([mutant_vector[j] if np.random.rand() < self.de_cross_prob or j == rand_idx else self.population[i][j] for j in range(self.dim)])
            self.population[i] = trial_vector

    def __call__(self, func):
        self.initialize_population()
        evaluations = 0
        while evaluations < self.budget:
            self.evaluate_population(func)
            evaluations += self.population_size
            self.differential_evolution_crossover()
            self.update_particles()
        return self.gbest_position