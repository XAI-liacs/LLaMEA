import numpy as np

class EnhancedAdaptiveMemorySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + self.dim
        self.memory_size = 5
        self.subpopulation_size = self.population_size // 2
        self.particles = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = np.inf
        self.omega = 0.5
        self.phi_p = 1.5
        self.phi_g = 1.5
        self.memory = []

    def _initialize_population(self, lb, ub):
        self.particles = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        self.velocities = np.random.rand(self.population_size, self.dim) * (ub - lb) / 10
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)

    def _update_particles(self, lb, ub):
        for i in range(self.population_size):
            # Select strategy with mutation
            if self.memory and np.random.rand() < 0.5:
                strategy = self.memory[np.random.randint(len(self.memory))]
                omega = strategy['omega'] + np.random.uniform(-0.05, 0.05)
                phi_p = strategy['phi_p'] + np.random.uniform(-0.05, 0.05)
                phi_g = strategy['phi_g'] + np.random.uniform(-0.05, 0.05)
            else:
                omega = self.omega
                phi_p = self.phi_p
                phi_g = self.phi_g

            cognitive_component = phi_p * np.random.rand(self.dim) * (self.personal_best_positions[i] - self.particles[i])
            social_component = phi_g * np.random.rand(self.dim) * (self.global_best_position - self.particles[i])
            self.velocities[i] = omega * self.velocities[i] + cognitive_component + social_component
            self.particles[i] += self.velocities[i]
            self.particles[i] = np.clip(self.particles[i], lb, ub)

    def _manage_subpopulations(self):
        subpopulations = np.array_split(self.particles, 2)
        for subpopulation in subpopulations:
            # Shuffle strategies within subpopulations
            np.random.shuffle(subpopulation)

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self._initialize_population(self.lb, self.ub)

        eval_count = 0
        while eval_count < self.budget:
            self._manage_subpopulations()

            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                score = func(self.particles[i])
                eval_count += 1

                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.particles[i]

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.particles[i]
                    if len(self.memory) >= self.memory_size:
                        self.memory.pop(0)
                    self.memory.append({'omega': self.omega, 'phi_p': self.phi_p, 'phi_g': self.phi_g})
                    self.omega = np.clip(self.omega + np.random.uniform(-0.05, 0.05), 0.3, 0.7)
                    self.phi_p = np.clip(self.phi_p + np.random.uniform(-0.05, 0.05), 1.2, 1.8)
                    self.phi_g = np.clip(self.phi_g + np.random.uniform(-0.05, 0.05), 1.2, 1.8)

            self._update_particles(self.lb, self.ub)

        return self.global_best_position, self.global_best_score