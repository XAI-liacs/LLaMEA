import numpy as np

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.f_weight = 0.8  # Weight for differential evolution
        self.c1 = 1.5  # Cognitive coefficient for PSO
        self.c2 = 1.5  # Social coefficient for PSO
        self.v_max = 0.2 * (self.upper_bound - self.lower_bound)

    def __call__(self, func):
        np.random.seed(42)
        pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-self.v_max, self.v_max, (self.population_size, self.dim))
        personal_best_positions = np.copy(pop)
        personal_best_scores = np.array([func(ind) for ind in pop])
        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]
        global_best_score = personal_best_scores[global_best_index]

        evals = self.population_size

        while evals < self.budget:
            # Differential Evolution
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = pop[a] + self.f_weight * (pop[b] - pop[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                crossover = np.random.rand(self.dim) < 0.95  # Changed crossover probability from 0.9 to 0.95
                trial = np.where(crossover, mutant, pop[i])
                trial_score = func(trial)
                evals += 1

                if trial_score < personal_best_scores[i]:
                    personal_best_positions[i] = trial
                    personal_best_scores[i] = trial_score

                    if trial_score < global_best_score:
                        global_best_position = trial
                        global_best_score = trial_score

            # Particle Swarm Optimization
            r1, r2 = np.random.rand(2, self.population_size, self.dim)
            velocities = (velocities + 
                          self.c1 * r1 * (personal_best_positions - pop) +
                          self.c2 * r2 * (global_best_position - pop))
            velocities = np.clip(velocities, -self.v_max, self.v_max)
            pop = pop + velocities
            pop = np.clip(pop, self.lower_bound, self.upper_bound)

            for i in range(self.population_size):
                score = func(pop[i])
                evals += 1

                if score < personal_best_scores[i]:
                    personal_best_positions[i] = pop[i]
                    personal_best_scores[i] = score

                    if score < global_best_score:
                        global_best_position = pop[i]
                        global_best_score = score

        return global_best_position, global_best_score