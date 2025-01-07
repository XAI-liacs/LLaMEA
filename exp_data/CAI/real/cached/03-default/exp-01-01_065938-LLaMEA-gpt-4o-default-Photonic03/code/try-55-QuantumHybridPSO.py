import numpy as np

class QuantumHybridPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.omega = 0.5
        self.phi_p = 0.5
        self.phi_g = 0.5
        self.mutation_rate = 0.1
    
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        position = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best = position.copy()
        personal_best_fitness = np.array([func(x) for x in position])
        global_best_idx = np.argmin(personal_best_fitness)
        global_best = personal_best[global_best_idx].copy()
        
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                r_p, r_g = np.random.rand(self.dim), np.random.rand(self.dim)
                velocity[i] = (self.omega * velocity[i] +
                               self.phi_p * r_p * (personal_best[i] - position[i]) +
                               self.phi_g * r_g * (global_best - position[i]))
                
                position[i] += velocity[i]
                position[i] = np.clip(position[i], lb, ub)
                
                # Apply quantum-inspired update
                quantum_position = position[i] + np.random.uniform(-1, 1, self.dim)
                quantum_position = np.clip(quantum_position, lb, ub)
                
                # Genetic-inspired mutation
                if np.random.rand() < self.mutation_rate:
                    mutation_index = np.random.randint(self.dim)
                    quantum_position[mutation_index] = np.random.uniform(lb[mutation_index], ub[mutation_index])
                
                fitness = func(quantum_position)
                evaluations += 1
                
                if fitness < personal_best_fitness[i]:
                    personal_best[i] = quantum_position
                    personal_best_fitness[i] = fitness
                    if fitness < personal_best_fitness[global_best_idx]:
                        global_best_idx = i
                        global_best = quantum_position
        
        return global_best