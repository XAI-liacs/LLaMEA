import numpy as np

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20  # You can adjust this size
        self.crossover_rate = 0.7
        self.differential_weight = 0.5
        self.inertia_weight = 0.5
        self.cognitive_coefficient = 1.5
        self.social_coefficient = 1.5

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = population.copy()
        personal_best_scores = np.full(self.population_size, np.inf)

        global_best_position = None
        global_best_score = np.inf

        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                # Evaluate individual
                current_score = func(population[i])
                evaluations += 1

                # Update personal best
                if current_score < personal_best_scores[i]:
                    personal_best_scores[i] = current_score
                    personal_best_positions[i] = population[i]

                # Update global best
                if current_score < global_best_score:
                    global_best_score = current_score
                    global_best_position = population[i]

            # DE mutation and crossover
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                # Select three random individuals
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[indices]

                # Mutation
                mutant = a + self.differential_weight * (b - c)
                mutant = np.clip(mutant, lb, ub)

                # Crossover
                trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, population[i])

                # Evaluate trial
                trial_score = func(trial)
                evaluations += 1

                # Selection
                if trial_score < current_score:
                    population[i] = trial

            # PSO update
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                # Update velocity
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_coefficient * r1 * (personal_best_positions[i] - population[i]) +
                                 self.social_coefficient * r2 * (global_best_position - population[i]))
                velocities[i] = np.clip(velocities[i], -1, 1)

                # Update position
                population[i] += velocities[i]
                population[i] = np.clip(population[i], lb, ub)

        return global_best_position, global_best_score