import numpy as np

class QuantumInspiredParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10 * dim // 2, 10)  # Modified line for adaptive population size
        self.alpha = 0.5  # Coefficient for quantum potential
        self.beta = 2.0   # Learning factor
        self.global_best_position = None

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

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Update velocities using quantum potential
                r1, r2 = np.random.rand(2)
                quantum_potential = self.alpha * r2 * (self.global_best_position - positions[i]) 
                self.beta = 1.5 + 0.5 * np.cos(np.pi * evaluations / self.budget)  # Modified line
                velocities[i] = r1 * velocities[i] + r2 * self.beta * (1 + 0.1 * np.cos(2 * np.pi * evaluations / self.budget)) * quantum_potential
                
                # Update positions based on new velocities
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], lower_bound, upper_bound)
                
                # Evaluate fitness
                fitness = func(positions[i])
                evaluations += 1
                
                # Update personal and global bests
                if fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_fitness[i] = fitness

                if fitness < global_best_fitness:
                    self.global_best_position = positions[i]
                    global_best_fitness = fitness
                
                if evaluations >= self.budget:
                    break

        return self.global_best_position, global_best_fitness