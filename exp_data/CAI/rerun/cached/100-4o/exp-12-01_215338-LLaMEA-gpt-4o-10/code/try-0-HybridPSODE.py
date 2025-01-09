import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.v_max = 0.2 * (self.upper_bound - self.lower_bound)
        self.v_min = -self.v_max
        self.c1 = 2.0  # cognitive coefficient
        self.c2 = 2.0  # social coefficient
        self.w = 0.7   # inertia weight
        self.mutation_factor = 0.5
        self.crossover_prob = 0.9

    def initialize_particles(self):
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(self.v_min, self.v_max, (self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.population_size, np.inf)
        return positions, velocities, personal_best_positions, personal_best_scores

    def de_mutation(self, population, best_idx):
        idxs = [idx for idx in range(self.population_size) if idx != best_idx]
        a, b, c = np.random.choice(idxs, 3, replace=False)
        mutant_vector = population[a] + self.mutation_factor * (population[b] - population[c])
        mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
        return mutant_vector

    def de_crossover(self, target, mutant):
        crossover = np.random.rand(self.dim) < self.crossover_prob
        trial_vector = np.where(crossover, mutant, target)
        return trial_vector

    def __call__(self, func):
        positions, velocities, personal_best_positions, personal_best_scores = self.initialize_particles()
        global_best_score = np.inf
        global_best_position = None

        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.population_size):
                # Evaluate fitness
                score = func(positions[i])
                evaluations += 1

                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]

                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i]

            # PSO velocity and position update
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (self.w * velocities + 
                          self.c1 * r1 * (personal_best_positions - positions) +
                          self.c2 * r2 * (global_best_position - positions))
            velocities = np.clip(velocities, self.v_min, self.v_max)
            positions += velocities
            positions = np.clip(positions, self.lower_bound, self.upper_bound)

            # DE Mutation and Crossover
            for i in range(self.population_size):
                mutant_vector = self.de_mutation(positions, i)
                trial_vector = self.de_crossover(positions[i], mutant_vector)
                
                trial_score = func(trial_vector)
                evaluations += 1
                
                if trial_score < personal_best_scores[i]:
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial_vector
                    if trial_score < global_best_score:
                        global_best_score = trial_score
                        global_best_position = trial_vector

            if evaluations >= self.budget:
                break
        
        return global_best_position, global_best_score