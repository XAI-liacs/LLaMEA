import numpy as np

class QuantumInspiredDynamicSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + self.dim
        self.memory_size = 5
        self.particles = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = np.inf
        self.alpha = 0.9  # Learning parameter
        self.beta = np.pi / 4  # Phase shift in quantum context
        self.memory = []

    def _initialize_population(self, lb, ub):
        self.particles = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        self.velocities = np.random.rand(self.population_size, self.dim) * (ub - lb) / 10
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)

    def _update_particles(self, lb, ub):
        for i in range(self.population_size):
            if self.memory and np.random.rand() < 0.5:
                strategy = self.memory[np.random.randint(len(self.memory))]
                alpha = strategy['alpha']
                beta = strategy['beta']
            else:
                alpha = self.alpha
                beta = self.beta

            phase_shift = np.cos(beta) * np.random.rand(self.dim)
            velocity_update = alpha * (self.personal_best_positions[i] - self.particles[i]) + phase_shift * (self.global_best_position - self.particles[i])
            self.velocities[i] = self.velocities[i] + velocity_update
            self.particles[i] += self.velocities[i]
            self.particles[i] = np.clip(self.particles[i], lb, ub)

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self._initialize_population(self.lb, self.ub)
        
        eval_count = 0
        while eval_count < self.budget:
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
                    self.memory.append({'alpha': self.alpha, 'beta': self.beta})

                    self.alpha = np.clip(self.alpha + np.random.uniform(-0.05, 0.05), 0.8, 1.0)
                    self.beta = np.clip(self.beta + np.random.uniform(-np.pi/16, np.pi/16), np.pi/8, np.pi/2)

            self._update_particles(self.lb, self.ub)

        return self.global_best_position, self.global_best_score