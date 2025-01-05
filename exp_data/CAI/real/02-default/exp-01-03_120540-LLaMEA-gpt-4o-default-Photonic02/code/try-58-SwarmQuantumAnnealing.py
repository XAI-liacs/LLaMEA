import numpy as np

class SwarmQuantumAnnealing:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.temperatures = np.linspace(1.0, 0.01, self.budget)
        self.positions = None
        self.velocities = None
        self.best_position = None
        self.best_score = float('inf')
        self.personal_best_positions = None
        self.personal_best_scores = None
        self.c1 = 1.5  # cognitive coefficient
        self.c2 = 1.5  # social coefficient
        self.inertia_weight = 0.7
    
    def initialize_swarm(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, float('inf'))
    
    def quantum_annealed_update(self, position, velocity, temperature, bounds):
        # Quantum-inspired perturbation using temperature
        perturbation = np.random.uniform(-1, 1, self.dim) * temperature
        new_position = position + velocity + perturbation
        return np.clip(new_position, bounds.lb, bounds.ub)
    
    def __call__(self, func):
        self.initialize_swarm(func.bounds)
        evaluations = 0
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                current_score = func(self.positions[i])
                evaluations += 1
                
                # Update personal best
                if current_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = current_score
                    self.personal_best_positions[i] = self.positions[i]
                
                # Update global best
                if current_score < self.best_score:
                    self.best_score = current_score
                    self.best_position = self.positions[i]
            
            # Temperature-based evolution
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                
                # Velocity update using swarm intelligence
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_component = self.c2 * r2 * (self.best_position - self.positions[i])
                self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                      cognitive_component + social_component)
                
                # Position update with quantum annealing
                temperature = self.temperatures[evaluations]
                self.positions[i] = self.quantum_annealed_update(self.positions[i], self.velocities[i], temperature, func.bounds)
                
                annealed_score = func(self.positions[i])
                evaluations += 1
                
                # Update global best with annealed score
                if annealed_score < self.best_score:
                    self.best_score = annealed_score
                    self.best_position = self.positions[i]