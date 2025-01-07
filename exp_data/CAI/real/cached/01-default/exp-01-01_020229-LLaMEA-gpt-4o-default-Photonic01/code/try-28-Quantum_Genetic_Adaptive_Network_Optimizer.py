import numpy as np

class Quantum_Genetic_Adaptive_Network_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.crossover_rate = 0.8
        self.mutation_rate = 0.1
        self.learning_rate = 0.1
        self.adaptive_decay = 0.99

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)

        position = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(p) for p in position])
        best_idx = np.argmin(fitness)
        global_best_position = np.copy(position[best_idx])
        global_best_value = fitness[best_idx]
        
        evaluations = self.population_size

        while evaluations < self.budget:
            new_population = np.copy(position)

            # Genetic crossover and mutation
            for i in range(self.population_size):
                if np.random.rand() < self.crossover_rate:
                    parent2 = position[np.random.randint(self.population_size)]
                    crossover_point = np.random.randint(self.dim)
                    new_population[i, :crossover_point] = position[i, :crossover_point]
                    new_population[i, crossover_point:] = parent2[crossover_point:]

                if np.random.rand() < self.mutation_rate:
                    mutation_indices = np.random.choice(self.dim, size=int(0.1 * self.dim), replace=False)
                    new_population[i, mutation_indices] += np.random.normal(0, 0.1, len(mutation_indices))
                    new_population[i] = np.clip(new_population[i], lb, ub)

            # Quantum-inspired update
            for i in range(self.population_size):
                phi = np.random.uniform(0, 2 * np.pi, self.dim)
                shift = self.learning_rate * (global_best_position - position[i]) * np.random.random(self.dim)
                position[i] += shift * np.sin(phi)
                position[i] = np.clip(position[i], lb, ub)

                current_value = func(position[i])
                evaluations += 1
                
                if current_value < fitness[i]:
                    new_population[i] = position[i]
                    fitness[i] = current_value
                
                if current_value < global_best_value:
                    global_best_position = position[i]
                    global_best_value = current_value

                if evaluations >= self.budget:
                    break

            # Adaptive learning rate decay
            self.learning_rate *= self.adaptive_decay
            position = new_population

        return global_best_position, global_best_value