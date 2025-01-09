import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim, population_size=40):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.w = 0.7  # inertia weight for PSO
        self.c1 = 1.5  # cognitive coefficient for PSO
        self.c2 = 1.5  # social coefficient for PSO
        self.mutation_factor = 0.5  # differential weight for DE
        self.crossover_prob = 0.7  # crossover probability for DE

    def __call__(self, func):
        np.random.seed(0)  # for reproducibility
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]
        global_best_score = personal_best_scores[global_best_index]
        eval_count = self.population_size

        while eval_count < self.budget:
            # PSO Update
            r1 = np.random.uniform(0, 1, (self.population_size, self.dim))
            r2 = np.random.uniform(0, 1, (self.population_size, self.dim))
            velocities = (self.w * velocities +
                          self.c1 * r1 * (personal_best_positions - population) +
                          self.c2 * r2 * (global_best_position - population))
            population += velocities
            population = np.clip(population, self.lower_bound, self.upper_bound)

            # DE Mutation and Crossover
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant_vector = np.clip(population[a] + self.mutation_factor * (population[b] - population[c]),
                                        self.lower_bound, self.upper_bound)
                crossover = np.random.rand(self.dim) < self.crossover_prob
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial_vector = np.where(crossover, mutant_vector, population[i])
                trial_score = func(trial_vector)
                eval_count += 1

                if trial_score < personal_best_scores[i]:
                    personal_best_positions[i] = trial_vector
                    personal_best_scores[i] = trial_score

                    if trial_score < global_best_score:
                        global_best_position = trial_vector
                        global_best_score = trial_score

                if eval_count >= self.budget:
                    break

        return global_best_position, global_best_score