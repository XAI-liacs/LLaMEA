import numpy as np

class QuantumPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.inertia_weight = 0.5
        self.cognitive_const = 1.5
        self.social_const = 1.5
        self.phi = 0.1  # Quantum potential factor

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        position = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        velocity = np.random.randn(self.population_size, self.dim) * 0.1
        p_best = np.copy(position)
        p_best_fitness = np.array([func(x) for x in position])
        
        g_best_idx = np.argmin(p_best_fitness)
        g_best = p_best[g_best_idx]
        
        evaluations = self.population_size
        
        while evaluations < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                cognitive_velocity = self.cognitive_const * r1 * (p_best[i] - position[i])
                social_velocity = self.social_const * r2 * (g_best - position[i])
                
                velocity[i] = (self.inertia_weight * velocity[i] + 
                               cognitive_velocity + 
                               social_velocity)
                
                # Quantum-inspired update
                potential = np.random.uniform(-self.phi, self.phi, self.dim)
                quantum_step = potential * (position[i] - g_best)
                position[i] = position[i] + velocity[i] + quantum_step
                position[i] = np.clip(position[i], lb, ub)
                
                fitness = func(position[i])
                evaluations += 1
                
                if fitness < p_best_fitness[i]:
                    p_best[i] = position[i]
                    p_best_fitness[i] = fitness
                    if fitness < p_best_fitness[g_best_idx]:
                        g_best_idx = i
                        g_best = position[i]

        return g_best