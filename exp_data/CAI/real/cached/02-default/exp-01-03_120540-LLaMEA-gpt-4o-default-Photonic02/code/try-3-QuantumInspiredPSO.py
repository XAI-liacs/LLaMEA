import numpy as np

class QuantumInspiredPSO:
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
        self.quantum_search_prob = 0.1  # Probability of performing quantum-inspired local search
        self.quantum_amplitude_initial = 0.1  # Initial amplitude for quantum exploration
    
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
        self.positions += self.velocities
    
    def quantum_inspired_search(self, func, particle_index, evaluations):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Adjust amplitude as evaluations increase
        amplitude = self.quantum_amplitude_initial * (1 - evaluations / self.budget)
        # Simulate quantum superposition with probabilistic local exploration
        quantum_pos = self.positions[particle_index] + np.random.choice([-1, 1], self.dim) * np.random.rand(self.dim) * amplitude
        quantum_pos = np.clip(quantum_pos, lb, ub)  # Ensure within bounds
        score = func(quantum_pos)
        return quantum_pos, score
    
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
                    
                # Perform quantum-inspired local search with some probability
                if np.random.rand() < self.quantum_search_prob and evaluations < self.budget:
                    quantum_pos, quantum_score = self.quantum_inspired_search(func, i, evaluations)
                    evaluations += 1
                    if quantum_score < self.personal_best_scores[i]:
                        self.personal_best_scores[i] = quantum_score
                        self.personal_best_positions[i] = quantum_pos
                    if quantum_score < self.global_best_score:
                        self.global_best_score = quantum_score
                        self.global_best_position = quantum_pos

            self.update_particles()
            self.inertia = 0.9 - 0.5 * (evaluations / self.budget)  # Adapt inertia over time