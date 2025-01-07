import numpy as np

class QuantumVariableSwarmExploration:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(15, dim * 2)
        self.alpha = 0.5  # Balance between exploration and exploitation
        self.beta = 0.1  # Quantum exploration rate
        self.gamma = 0.7  # Swarm update influence
        self.momentum = 0.6  # Initial momentum

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim)) * (ub - lb)
        scores = np.array([func(individual) for individual in population])
        evaluations = self.population_size
        
        best_index = np.argmin(scores)
        best_position = population[best_index].copy()
        
        while evaluations < self.budget:
            new_population = np.empty_like(population)
            for i in range(self.population_size):
                r1 = np.random.rand(self.dim)
                rand_quantum = np.random.normal(0, self.beta, self.dim) * (ub - lb)
                
                # Quantum-inspired variable exploration
                quantum_jump = best_position + rand_quantum
                quantum_jump = np.clip(quantum_jump, lb, ub)
                
                if np.random.rand() < self.alpha:
                    new_population[i] = quantum_jump
                else:
                    # Swarm dynamics update
                    inertia = self.momentum * velocities[i]
                    cognitive = self.gamma * r1 * (best_position - population[i])
                    
                    velocities[i] = inertia + cognitive
                    new_population[i] = population[i] + velocities[i]
                    new_population[i] = np.clip(new_population[i], lb, ub)
            
            new_scores = np.array([func(individual) for individual in new_population])
            evaluations += self.population_size
            
            # Update the best found solution
            current_best_index = np.argmin(new_scores)
            if new_scores[current_best_index] < scores[best_index]:
                best_index = current_best_index
                best_position = new_population[best_index].copy()

            # Keep the new population and scores
            population, scores = new_population, new_scores
        
        return best_position, scores[best_index]