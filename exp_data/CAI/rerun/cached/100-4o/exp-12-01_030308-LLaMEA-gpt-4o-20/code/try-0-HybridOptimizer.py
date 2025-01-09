import numpy as np

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.inertia_weight = 0.7
        self.cognitive_constant = 1.5
        self.social_constant = 1.5
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7

    def __call__(self, func):
        np.random.seed(0)
        pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(pop)
        personal_best_scores = np.array([func(ind) for ind in pop])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Particle Swarm Optimization update
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_constant * r1 * (personal_best_positions[i] - pop[i]) +
                                 self.social_constant * r2 * (global_best_position - pop[i]))

                pop[i] = np.clip(pop[i] + velocities[i], self.lower_bound, self.upper_bound)
                score = func(pop[i])
                evaluations += 1

                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = pop[i]

                # Differential Evolution crossover and mutation
                if evaluations < self.budget:
                    a, b, c = pop[np.random.choice(self.population_size, 3, replace=False)]
                    mutant_vector = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                    trial_vector = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant_vector, pop[i])
                    trial_score = func(trial_vector)
                    evaluations += 1

                    if trial_score < score:
                        pop[i] = trial_vector
                        score = trial_score

                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = pop[i]

        return global_best_position, global_best_score