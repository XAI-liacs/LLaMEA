import numpy as np

class QuantumGeneticPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, dim)
        self.inertia = 0.7
        self.c1 = 1.5
        self.c2 = 1.5
        self.mutation_rate = 0.1
        self.crossover_rate = 0.7
        self.beta = 0.05

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim)) * (ub - lb)
        personal_best_positions = population.copy()
        personal_best_scores = np.array([func(population[i]) for i in range(self.population_size)])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = population[global_best_index].copy()
        evaluations = self.population_size

        while evaluations < self.budget:
            # Crossover and Mutation (Quantum-inspired Genetic Operations)
            new_population = np.empty((self.population_size, self.dim))
            for i in range(self.population_size):
                if np.random.rand() < self.crossover_rate:
                    partner_index = np.random.randint(self.population_size)
                    crossover_point = np.random.randint(1, self.dim)
                    new_population[i, :crossover_point] = personal_best_positions[i, :crossover_point]
                    new_population[i, crossover_point:] = personal_best_positions[partner_index, crossover_point:]
                else:
                    new_population[i] = personal_best_positions[i]
                
                if np.random.rand() < self.mutation_rate:
                    mutation_vector = np.random.normal(0, 1, self.dim) * (ub - lb)
                    new_population[i] += mutation_vector * self.beta
                    new_population[i] = np.clip(new_population[i], lb, ub)

            # PSO velocity and position update
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (self.inertia * velocities +
                          self.c1 * r1 * (personal_best_positions - population) +
                          self.c2 * r2 * (global_best_position - population))
            population = np.clip(population + velocities, lb, ub)

            # Evaluate new population
            new_scores = np.array([func(new_population[i]) for i in range(self.population_size)])
            evaluations += self.population_size

            # Update personal and global best
            for i in range(self.population_size):
                if new_scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = new_scores[i]
                    personal_best_positions[i] = new_population[i]

            current_global_best_index = np.argmin(personal_best_scores)
            if personal_best_scores[current_global_best_index] < personal_best_scores[global_best_index]:
                global_best_index = current_global_best_index
                global_best_position = personal_best_positions[global_best_index]

        # Return the best solution found
        return global_best_position, personal_best_scores[global_best_index]