import numpy as np

class AdaptiveQuantumEvolutionary:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(20, 2 * dim)
        self.beta = 0.1   # Quantum-inspired learning rate
        self.mutation_rate = 0.2
        self.crossover_rate = 0.9
        self.inertia = 0.5  # Initial inertia weight
        self.cognitive_coeff = 1.0
        self.social_coeff = 1.2

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim)) * (ub - lb)
        personal_best_positions = population.copy()
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index].copy()
        evaluations = self.population_size

        while evaluations < self.budget:
            # Adaptive inertia weight
            inertia = self.inertia + 0.2 * (1 - evaluations / self.budget)

            # Evolutionary Crossover and Mutation
            for i in range(self.population_size):
                if np.random.rand() < self.crossover_rate:
                    parents = np.random.choice(self.population_size, 2, replace=False)
                    crossover_point = np.random.randint(1, self.dim)
                    population[i, :crossover_point] = personal_best_positions[parents[0], :crossover_point]
                    population[i, crossover_point:] = personal_best_positions[parents[1], crossover_point:]

                if np.random.rand() < self.mutation_rate:
                    mutation_vector = np.random.normal(0, 0.1, self.dim) * (ub - lb)
                    population[i] = np.clip(population[i] + mutation_vector, lb, ub)

            # Quantum-inspired PSO update
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (inertia * velocities +
                          self.cognitive_coeff * r1 * (personal_best_positions - population) +
                          self.social_coeff * r2 * (global_best_position - population))
            population = np.clip(population + velocities, lb, ub)

            # Quantum-inspired exploration
            if np.random.rand() < self.beta:
                q = np.random.normal(loc=0, scale=1, size=self.dim)
                new_particle = global_best_position + q * (ub - lb)
                population[np.random.randint(self.population_size)] = np.clip(new_particle, lb, ub)

            # Evaluate the new population
            scores = np.array([func(ind) for ind in population])
            evaluations += self.population_size

            # Update personal and global bests
            improved = scores < personal_best_scores
            personal_best_scores[improved] = scores[improved]
            personal_best_positions[improved] = population[improved]

            global_best_index = np.argmin(personal_best_scores)
            global_best_position = personal_best_positions[global_best_index].copy()

        # Return the best solution found
        return global_best_position, personal_best_scores[global_best_index]