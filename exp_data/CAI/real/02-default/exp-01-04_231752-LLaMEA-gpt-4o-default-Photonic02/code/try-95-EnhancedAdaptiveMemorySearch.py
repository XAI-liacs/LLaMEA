import numpy as np

class EnhancedAdaptiveMemorySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + self.dim
        self.short_memory_size = 5  # Short-term memory size
        self.long_memory_size = 10  # Long-term memory size
        self.particles = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = np.inf
        self.omega = 0.5  # Inertia weight
        self.phi_p = 1.5  # Cognitive coefficient
        self.phi_g = 1.5  # Social coefficient
        self.short_memory = []  # Short-term memory
        self.long_memory = []   # Long-term memory

    def _initialize_population(self, lb, ub):
        self.particles = np.random.rand(self.population_size, self.dim) * (ub - lb) + lb
        self.velocities = np.random.rand(self.population_size, self.dim) * (ub - lb) / 10
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)

    def _update_particles(self, lb, ub):
        for i in range(self.population_size):
            # Choose strategy from short or long-term memory
            if self.short_memory and np.random.rand() < 0.3:
                strategy = self.short_memory[np.random.randint(len(self.short_memory))]
            elif self.long_memory and np.random.rand() < 0.3:
                strategy = self.long_memory[np.random.randint(len(self.long_memory))]
            else:
                strategy = {'omega': self.omega, 'phi_p': self.phi_p, 'phi_g': self.phi_g}
                
            cognitive_component = strategy['phi_p'] * np.random.rand(self.dim) * (self.personal_best_positions[i] - self.particles[i])
            social_component = strategy['phi_g'] * np.random.rand(self.dim) * (self.global_best_position - self.particles[i])
            inertia = strategy['omega']

            self.velocities[i] = inertia * self.velocities[i] + cognitive_component + social_component
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
                    
                    # Add successful strategy to both memories
                    if len(self.short_memory) >= self.short_memory_size:
                        self.short_memory.pop(0)
                    self.short_memory.append({'omega': self.omega, 'phi_p': self.phi_p, 'phi_g': self.phi_g})

                    if len(self.long_memory) >= self.long_memory_size:
                        self.long_memory.pop(0)
                    self.long_memory.append({'omega': self.omega, 'phi_p': self.phi_p, 'phi_g': self.phi_g})

                    # Adapt parameters slightly
                    self.omega = np.clip(self.omega + np.random.uniform(-0.05, 0.05), 0.3, 0.7)
                    self.phi_p = np.clip(self.phi_p + np.random.uniform(-0.05, 0.05), 1.2, 1.8)
                    self.phi_g = np.clip(self.phi_g + np.random.uniform(-0.05, 0.05), 1.2, 1.8)

            if eval_count / self.budget > 0.5:  # Increase population size after half of the budget
                self.population_size = min(self.population_size + 5, int(1.5 * (20 + self.dim)))
                self._initialize_population(self.lb, self.ub)

            self._update_particles(self.lb, self.ub)

        return self.global_best_position, self.global_best_score