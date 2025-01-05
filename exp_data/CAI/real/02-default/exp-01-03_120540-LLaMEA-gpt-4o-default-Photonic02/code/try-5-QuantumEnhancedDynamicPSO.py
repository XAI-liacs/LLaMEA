import numpy as np

class QuantumEnhancedDynamicPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 20
        self.positions = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.inertia = 0.9
        self.cognitive_coeff = 2.0
        self.social_coeff = 2.0
        self.local_search_prob = 0.1  # Probability of performing local search
        self.initial_local_search_radius = 0.1  # Initial radius for local search exploration
    
    def initialize_particles(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        self.velocities = np.random.uniform(-abs(ub - lb), abs(ub - lb), (self.num_particles, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.num_particles, float('inf'))
        
    def update_particles(self):
        r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
        cognitive_component = self.cognitive_coeff * r1 * (self.personal_best_positions - self.positions)
        social_component = self.social_coeff * r2 * (self.global_best_position - self.positions)
        self.velocities = self.inertia * self.velocities + cognitive_component + social_component
        quantum_particles = self.positions + np.random.uniform(-1, 1, self.positions.shape) * self.velocities
        self.positions = np.clip(quantum_particles, func.bounds.lb, func.bounds.ub)
    
    def adaptive_local_search(self, func, particle_index, evaluations):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Decrease the local search radius dynamically as evaluations increase
        local_search_radius = self.initial_local_search_radius * (1 - evaluations / self.budget)
        local_pos = self.positions[particle_index] + np.random.uniform(-local_search_radius, local_search_radius, self.dim)
        local_pos = np.clip(local_pos, lb, ub)  # Ensure within bounds
        score = func(local_pos)
        return local_pos, score
    
    def __call__(self, func):
        self.initialize_particles(func.bounds)
        evaluations = 0
        
        while evaluations < self.budget:
            for i in range(self.num_particles):
                if evaluations >= self.budget:
                    break

                score = func(self.positions[i])
                evaluations += 1

                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i]
                
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i]
                    
                # Perform local search with some probability
                if np.random.rand() < self.local_search_prob and evaluations < self.budget:
                    local_pos, local_score = self.adaptive_local_search(func, i, evaluations)
                    evaluations += 1
                    if local_score < self.personal_best_scores[i]:
                        self.personal_best_scores[i] = local_score
                        self.personal_best_positions[i] = local_pos
                    if local_score < self.global_best_score:
                        self.global_best_score = local_score
                        self.global_best_position = local_pos

            self.update_particles()
            # Adapt inertia and coefficients over time for better exploration-exploitation balance
            self.inertia = 0.9 - 0.5 * (evaluations / self.budget)
            self.cognitive_coeff = 2.0 - 1.0 * (evaluations / self.budget)
            self.social_coeff = 2.0 + 1.0 * (evaluations / self.budget)