import numpy as np

class AdaptiveQuantumParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.current_evaluations = 0
        self.phi = 0.5 + np.log(2)  # Golden ratio for attraction
        self.alpha = 0.5  # Constriction factor
        self.beta = 0.5  # Quantum position shift factor
        
    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.rand(self.population_size, self.dim) * (bounds[:,1] - bounds[:,0]) + bounds[:,0]
        velocities = np.random.rand(self.population_size, self.dim) * (bounds[:,1] - bounds[:,0]) / 20
        personal_best_positions = np.copy(population)
        personal_best_fitness = np.array([func(ind) for ind in population])
        global_best_index = np.argmin(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_index]
        self.current_evaluations += self.population_size
        
        while self.current_evaluations < self.budget:
            for i in range(self.population_size):
                neighborhood_best = np.copy(personal_best_positions[i])
                if i > 0:
                    neighborhood_best = min(neighborhood_best, personal_best_positions[i-1], key=func)
                if i < self.population_size - 1:
                    neighborhood_best = min(neighborhood_best, personal_best_positions[i+1], key=func)
                
                velocities[i] = self.alpha * velocities[i] + \
                                np.random.rand(self.dim) * (personal_best_positions[i] - population[i]) + \
                                np.random.rand(self.dim) * (neighborhood_best - population[i])
                population[i] = population[i] + velocities[i]
                population[i] = np.clip(population[i], bounds[:,0], bounds[:,1])
                
                current_fitness = func(population[i])
                self.current_evaluations += 1
                
                if current_fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = population[i]
                    personal_best_fitness[i] = current_fitness
                    if current_fitness < personal_best_fitness[global_best_index]:
                        global_best_position = population[i]
                        global_best_index = i
            
            # Quantum swarm re-positioning
            if self.current_evaluations < self.budget:
                for i in range(self.population_size):
                    self.beta = 0.5 + (0.5 * np.sin(self.current_evaluations/self.budget * np.pi))  # Sine based beta
                    quantum_shift = self.beta * (np.random.rand(self.dim) - 0.5)
                    quantum_position = global_best_position + quantum_shift
                    quantum_position = np.clip(quantum_position, bounds[:,0], bounds[:,1])
                    quantum_fitness = func(quantum_position)
                    self.current_evaluations += 1
                    
                    if quantum_fitness < personal_best_fitness[i]:
                        personal_best_positions[i] = quantum_position
                        personal_best_fitness[i] = quantum_fitness
                        if quantum_fitness < personal_best_fitness[global_best_index]:
                            global_best_position = quantum_position
                            global_best_index = i

            if self.current_evaluations >= self.budget:
                break

        return global_best_position, personal_best_fitness[global_best_index]