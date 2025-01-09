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
                self.alpha = 0.3 + 0.2 * (self.current_evaluations/self.budget)  # Dynamic inertia
                c1 = 1.5  # Personal acceleration coefficient
                c2 = 1.5 + (self.current_evaluations/self.budget)  # Dynamic social acceleration
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = self.alpha * velocities[i] + \
                                c1 * r1 * (personal_best_positions[i] - population[i]) + \
                                c2 * r2 * (global_best_position - population[i])
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

            # Local search around global best
            local_perturbation = np.random.normal(scale=0.1 * (1 - self.current_evaluations/self.budget), size=self.dim)  # Dynamic perturbation
            candidate_position = np.clip(global_best_position + local_perturbation, bounds[:,0], bounds[:,1])
            candidate_fitness = func(candidate_position)
            self.current_evaluations += 1
            
            if candidate_fitness < personal_best_fitness[global_best_index]:
                global_best_position = candidate_position
                global_best_index = np.argmin(personal_best_fitness)
            
            self.population_size = max(1, int(10 * self.dim * (1 - self.current_evaluations / self.budget)))

            if self.current_evaluations >= self.budget:
                break

        return global_best_position, personal_best_fitness[global_best_index]