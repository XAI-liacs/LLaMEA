import numpy as np

class DynamicStrategyExploration:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + self.dim
        self.memory_size = 5  # Number of successful strategies to remember
        self.strategy_pool_size = 10  # Pool of diverse strategies
        self.particles = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = np.inf
        self.omega = 0.5  # Inertia weight
        self.phi_p = 1.5  # Cognitive coefficient
        self.phi_g = 1.5  # Social coefficient
        self.memory = []  # Initialize adaptive memory
        self.strategy_pool = self._initialize_strategy_pool()

    def _initialize_strategy_pool(self):
        pool = []
        for _ in range(self.strategy_pool_size):
            strategy = {
                'omega': np.random.uniform(0.3, 0.7),
                'phi_p': np.random.uniform(1.2, 1.8),
                'phi_g': np.random.uniform(1.2, 1.8)
            }
            pool.append(strategy)
        return pool

    def _initialize_population(self, lb, ub):
        self.particles = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        self.velocities = np.random.rand(self.population_size, self.dim) * (ub - lb) / 10
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)

    def _update_particles(self, lb, ub):
        for i in range(self.population_size):
            # Choose a strategy from memory or explore the strategy pool
            if self.memory and np.random.rand() < 0.5:
                strategy = self.memory[np.random.randint(len(self.memory))]
            else:
                strategy = self.strategy_pool[np.random.randint(len(self.strategy_pool))]

            cognitive_component = strategy['phi_p'] * np.random.rand(self.dim) * (self.personal_best_positions[i] - self.particles[i])
            social_component = strategy['phi_g'] * np.random.rand(self.dim) * (self.global_best_position - self.particles[i])
            inertia = strategy['omega']

            self.velocities[i] = inertia * self.velocities[i] + cognitive_component + social_component
            perturbation = 0.05 * np.random.randn(self.dim)  # Small random perturbation for exploration
            self.particles[i] = np.clip(self.particles[i] + self.velocities[i] + perturbation, lb, ub)

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
                    # Add successful strategy to memory
                    if len(self.memory) >= self.memory_size:
                        self.memory.pop(0)  # Remove oldest strategy if memory is full
                    self.memory.append({'omega': self.omega, 'phi_p': self.phi_p, 'phi_g': self.phi_g})
                    # Adapt parameters slightly and update strategy pool
                    self.omega = np.clip(self.omega + np.random.uniform(-0.1, 0.1), 0.3, 0.7)
                    self.phi_p = np.clip(self.phi_p + np.random.uniform(-0.1, 0.1), 1.2, 1.8)
                    self.phi_g = np.clip(self.phi_g + np.random.uniform(-0.1, 0.1), 1.2, 1.8)
                    self.strategy_pool[np.random.randint(self.strategy_pool_size)] = {'omega': self.omega, 'phi_p': self.phi_p, 'phi_g': self.phi_g}

            self._update_particles(self.lb, self.ub)

        return self.global_best_position, self.global_best_score