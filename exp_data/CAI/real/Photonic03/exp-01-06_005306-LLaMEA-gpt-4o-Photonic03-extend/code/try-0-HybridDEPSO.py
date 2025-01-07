import numpy as np

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.8  # Scaling factor for DE
        self.Cr = 0.9  # Crossover probability for DE
        self.w = 0.5  # Inertia weight for PSO
        self.c1 = 1.5  # Cognitive coefficient for PSO
        self.c2 = 1.5  # Social coefficient for PSO
        self.global_best_position = None
        self.global_best_value = np.inf

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        personal_best_positions = np.copy(pop)
        personal_best_values = np.array([func(ind) for ind in pop])

        if np.min(personal_best_values) < self.global_best_value:
            self.global_best_value = np.min(personal_best_values)
            self.global_best_position = personal_best_positions[np.argmin(personal_best_values)]

        evaluations = self.population_size

        while evaluations < self.budget:
            # Hybrid DE/PSO iteration
            for i in range(self.population_size):
                a, b, c = np.random.choice([j for j in range(self.population_size) if j != i], 3, replace=False)
                mutant = pop[a] + self.F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)

                trial = np.copy(pop[i])
                for j in range(self.dim):
                    if np.random.rand() < self.Cr:
                        trial[j] = mutant[j]

                trial_value = func(trial)
                evaluations += 1

                if trial_value < personal_best_values[i]:
                    personal_best_values[i] = trial_value
                    personal_best_positions[i] = trial

                    if trial_value < self.global_best_value:
                        self.global_best_value = trial_value
                        self.global_best_position = trial

                # PSO Update
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = self.w * velocities[i] + \
                                self.c1 * r1 * (personal_best_positions[i] - pop[i]) + \
                                self.c2 * r2 * (self.global_best_position - pop[i])
                pop[i] = np.clip(pop[i] + velocities[i], lb, ub)

            # Adaptive parameter tuning
            self.F = 0.5 + np.random.rand() * 0.5
            self.Cr = 0.5 + np.random.rand() * 0.5
            self.w = 0.4 + np.random.rand() * 0.5

        return self.global_best_position, self.global_best_value