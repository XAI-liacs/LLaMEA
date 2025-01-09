import numpy as np

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocity = np.random.uniform(-1, 1, self.population.shape)
        self.best_positions = self.population.copy()
        self.best_fitness = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_fitness = np.inf

    def __call__(self, func):
        evaluations = 0
        
        while evaluations < self.budget:
            # Evaluate fitness
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                fitness = func(self.population[i])
                evaluations += 1
                if fitness < self.best_fitness[i]:
                    self.best_fitness[i] = fitness
                    self.best_positions[i] = self.population[i].copy()
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = self.population[i].copy()

            # Update velocities and positions using a DE/PSO hybrid approach
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                inertia_component = 0.5 * self.velocity[i]
                cognitive_component = 1.5 * r1 * (self.best_positions[i] - self.population[i])
                social_component = 1.5 * r2 * (self.global_best_position - self.population[i])
                
                # Differential evolution mutation step
                idxs = np.random.choice(self.population_size, 3, replace=False)
                while i in idxs:
                    idxs = np.random.choice(self.population_size, 3, replace=False)
                mutant_vector = self.population[idxs[0]] + 0.8 * (self.population[idxs[1]] - self.population[idxs[2]])
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)

                # Combine DE mutant with PSO velocity update
                if np.random.rand() < 0.5:
                    self.velocity[i] = inertia_component + cognitive_component + social_component
                else:
                    self.velocity[i] = mutant_vector - self.population[i]
                
                self.population[i] = self.population[i] + self.velocity[i]
                self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)

        return self.global_best_position, self.global_best_fitness