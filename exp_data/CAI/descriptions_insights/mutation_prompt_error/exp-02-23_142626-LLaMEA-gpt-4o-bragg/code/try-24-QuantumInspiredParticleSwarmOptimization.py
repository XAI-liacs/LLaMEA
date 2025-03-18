import numpy as np

class QuantumInspiredParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.alpha = 0.5  # Coefficient for quantum potential
        self.beta = 2.0   # Learning factor
        self.global_best_position = None
        self.inertia_weight = 0.9  # Adaptive inertia weight
    
    def __call__(self, func):
        lower_bound = func.bounds.lb
        upper_bound = func.bounds.ub
        
        # Initialize particles' positions and velocities
        positions = np.random.uniform(lower_bound, upper_bound, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        personal_best_positions = positions.copy()
        personal_best_fitness = np.array([func(ind) for ind in positions])
        evaluations = self.population_size
        
        if self.global_best_position is None:
            best_index = np.argmin(personal_best_fitness)
            self.global_best_position = personal_best_positions[best_index]
            global_best_fitness = personal_best_fitness[best_index]

        convergence_threshold = 1e-5  # Threshold for convergence improvement
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                quantum_potential = self.alpha * r2 * (self.global_best_position - positions[i])
                self.beta = 1.5 + 0.5 * np.sin(np.pi * evaluations / self.budget)
                self.inertia_weight = 0.4 + 0.5 * (1 - evaluations / self.budget)  # Update inertia
                velocities[i] = self.inertia_weight * velocities[i] + r2 * self.beta * quantum_potential
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], lower_bound, upper_bound)
                
                fitness = func(positions[i])
                evaluations += 1
                
                if fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_fitness[i] = fitness

                if fitness < global_best_fitness:
                    self.global_best_position = positions[i]
                    global_best_fitness = fitness
                
                if evaluations >= self.budget or abs(global_best_fitness - fitness) < convergence_threshold:
                    break

        return self.global_best_position, global_best_fitness