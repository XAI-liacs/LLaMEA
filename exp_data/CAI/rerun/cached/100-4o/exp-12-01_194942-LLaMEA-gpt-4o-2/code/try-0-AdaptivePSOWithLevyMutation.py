import numpy as np

class AdaptivePSOWithLevyMutation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.c1 = 2.0
        self.c2 = 2.0
        self.w = 0.7
        self.velocity_max = 0.2 * (self.upper_bound - self.lower_bound)

    def levy_flight(self, L):
        # Levy distribution
        return np.random.normal(0, 1, L) / (np.abs(np.random.normal(0, 1, L))**(1/3))

    def __call__(self, func):
        np.random.seed(42)  # For reproducibility
        pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-self.velocity_max, self.velocity_max, (self.population_size, self.dim))
        personal_best_positions = np.copy(pop)
        personal_best_scores = np.array([func(ind) for ind in pop])

        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]
        global_best_score = personal_best_scores[global_best_index]

        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - pop[i]) +
                                 self.c2 * r2 * (global_best_position - pop[i]))

                velocities[i] = np.clip(velocities[i], -self.velocity_max, self.velocity_max)
                pop[i] += velocities[i]
                pop[i] = np.clip(pop[i], self.lower_bound, self.upper_bound)

                if np.random.rand() < 0.1:  # Apply Levy mutation with 10% probability
                    pop[i] += self.levy_flight(self.dim)

                score = func(pop[i])
                eval_count += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = pop[i]

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = pop[i]

                if eval_count >= self.budget:  # Stop if budget is exhausted
                    break

        return global_best_position, global_best_score