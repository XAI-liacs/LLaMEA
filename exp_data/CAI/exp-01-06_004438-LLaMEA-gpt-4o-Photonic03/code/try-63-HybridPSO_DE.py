import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.c1 = 1.49445
        self.c2 = 1.49445
        self.w = 0.729
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))

        personal_best = population.copy()
        personal_best_value = np.array([func(ind) for ind in population])
        global_best = personal_best[np.argmin(personal_best_value)]
        global_best_value = np.min(personal_best_value)

        evals = self.population_size
        while evals < self.budget:
            # PSO Update with dynamic inertia weight
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            self.w = 0.9 - (0.4 * evals / self.budget) + np.random.normal(0, 0.03)  # Modified inertia weight adjustment
            self.c1 = 1.49445 + 0.5 * (evals / self.budget)  # Adaptive acceleration coefficient for personal best
            dynamic_personal_factor = 0.6 + 0.4 * (1 - evals / self.budget)  # Updated dynamic personal influence factor
            velocity = (self.w * velocity + 
                        dynamic_personal_factor * self.c1 * r1 * (personal_best - population) + 
                        self.c2 * r2 * (global_best - population))  # Adjust dynamic personal factor here
            population = population + velocity
            population = np.clip(population, lb, ub)

            # DE Update
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                dynamic_factor = 0.5 + 0.5 * (1 - evals / self.budget)  # New dynamic scaling factor
                mutant = np.clip(population[a] + dynamic_factor * (population[b] - population[c]), lb, ub)
                self.crossover_rate = 0.9 * np.random.rand()  # Introduce stochastic crossover rate
                cross_points = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Evaluate trial vector
                trial_value = func(trial)
                evals += 1

                # Selection
                if trial_value < personal_best_value[i]:
                    personal_best[i] = trial
                    personal_best_value[i] = trial_value

            # Update global best
            min_index = np.argmin(personal_best_value)
            if personal_best_value[min_index] < global_best_value:
                global_best = personal_best[min_index]
                global_best_value = personal_best_value[min_index]

            if evals >= self.budget:
                break

        return global_best, global_best_value