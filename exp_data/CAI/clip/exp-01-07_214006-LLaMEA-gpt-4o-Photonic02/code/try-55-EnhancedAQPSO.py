import numpy as np

class EnhancedAQPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.current_evaluations = 0
        self.phi = 0.5 + np.log(2)
        self.alpha = 0.5
        self.beta = 0.5

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
            diversity_factor = np.std(population, axis=0).mean()
            for i in range(self.population_size):
                self.alpha = 0.3 + 0.2 * (self.current_evaluations/self.budget)
                c1 = 1.5
                c2 = 1.5 + (self.current_evaluations/self.budget)
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = self.alpha * velocities[i] + \
                                c1 * r1 * (personal_best_positions[i] - population[i]) + \
                                c2 * r2 * (global_best_position - population[i])
                population[i] = np.clip(population[i] + velocities[i], bounds[:,0], bounds[:,1])
                
                current_fitness = func(population[i])
                self.current_evaluations += 1
                
                if current_fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = population[i]
                    personal_best_fitness[i] = current_fitness
                    if current_fitness < personal_best_fitness[global_best_index]:
                        global_best_position = population[i]
                        global_best_index = i

            if diversity_factor < 1e-3:
                for i in range(self.population_size):
                    candidates = np.random.choice(self.population_size, 3, replace=False)
                    mutant_vector = (population[candidates[0]] + 
                                     0.5 * (population[candidates[1]] - population[candidates[2]]))
                    mutant_vector = np.clip(mutant_vector, bounds[:,0], bounds[:,1])
                    mutant_fitness = func(mutant_vector)
                    self.current_evaluations += 1
                    if mutant_fitness < personal_best_fitness[i]:
                        personal_best_positions[i] = mutant_vector
                        personal_best_fitness[i] = mutant_fitness
                        if mutant_fitness < personal_best_fitness[global_best_index]:
                            global_best_position = mutant_vector
                            global_best_index = i

            if self.current_evaluations >= self.budget:
                break

        return global_best_position, personal_best_fitness[global_best_index]