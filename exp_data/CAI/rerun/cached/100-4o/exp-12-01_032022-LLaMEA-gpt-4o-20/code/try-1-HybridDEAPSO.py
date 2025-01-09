import numpy as np

class HybridDEAPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.c1 = 1.5  # PSO cognitive coefficient
        self.c2 = 1.5  # PSO social coefficient
        self.w = 0.7   # Inertia weight for PSO
        self.func_eval = 0

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, 
                                       (self.population_size, self.dim))
        velocities = np.random.uniform(-0.1, 0.1, (self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.full(self.population_size, np.inf)
        global_best_position = None
        global_best_score = np.inf

        while self.func_eval < self.budget:
            for i in range(self.population_size):
                if self.func_eval >= self.budget:
                    break
                
                # Evaluate the objective function
                fitness = func(population[i])
                self.func_eval += 1

                # Update personal best
                if fitness < personal_best_scores[i]:
                    personal_best_scores[i] = fitness
                    personal_best_positions[i] = population[i]

                # Update global best
                if fitness < global_best_score:
                    global_best_score = fitness
                    global_best_position = population[i]

            # Hybrid: Differential Evolution Mutation and Crossover
            for i in range(self.population_size):
                if self.func_eval >= self.budget:
                    break

                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                
                mutant = population[a] + self.mutation_factor * (population[b] - population[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                
                trial = np.copy(population[i])
                crossover = np.random.rand(self.dim) < self.crossover_rate
                trial[crossover] = mutant[crossover]
                
                trial_fitness = func(trial)
                self.func_eval += 1

                if trial_fitness < fitness:
                    population[i] = trial
                    fitness = trial_fitness

            # Adaptive PSO velocity and position update
            for i in range(self.population_size):
                if self.func_eval >= self.budget:
                    break

                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - population[i]) +
                                 self.c2 * r2 * (global_best_position - population[i]))
                
                population[i] = population[i] + velocities[i]
                population[i] = np.clip(population[i], self.lower_bound, self.upper_bound)

        return global_best_position, global_best_score