import numpy as np

class EnsembleOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.crossover_rate = 0.7
        self.mutation_factor = 0.8
        self.inertia_weight = 0.5
        self.cognitive_coefficient = 2.0
        self.social_coefficient = 2.0

    def __call__(self, func):
        # Initialize population for DE
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        # Initialize velocities and personal bests for PSO
        velocities = np.random.uniform(-0.1, 0.1, (self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in population])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]

        num_evaluations = self.population_size

        while num_evaluations < self.budget:
            # Differential Evolution strategy
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant_vector = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                crossover = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(crossover):
                    crossover[np.random.randint(0, self.dim)] = True
                trial_vector = np.where(crossover, mutant_vector, population[i])
                trial_score = func(trial_vector)
                num_evaluations += 1
                if trial_score < personal_best_scores[i]:
                    personal_best_scores[i] = trial_score
                    personal_best_positions[i] = trial_vector
                    if trial_score < personal_best_scores[global_best_index]:
                        global_best_index = i
                        global_best_position = trial_vector

            # Particle Swarm Optimization strategy
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2, self.dim)
                cognitive_velocity = self.cognitive_coefficient * r1 * (personal_best_positions[i] - population[i])
                social_velocity = self.social_coefficient * r2 * (global_best_position - population[i])
                velocities[i] = (self.inertia_weight * velocities[i] + cognitive_velocity + social_velocity)
                population[i] = np.clip(population[i] + velocities[i], self.lower_bound, self.upper_bound)
                score = func(population[i])
                num_evaluations += 1
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = population[i]
                    if score < personal_best_scores[global_best_index]:
                        global_best_index = i
                        global_best_position = population[i]

        return global_best_position

# Example usage:
# optimizer = EnsembleOptimizer(budget=1000, dim=10)
# best_solution = optimizer(some_black_box_function)