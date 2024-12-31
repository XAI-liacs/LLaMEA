import numpy as np

class DE_PSO_Hybrid:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.f_opt = np.Inf
        self.x_opt = None
        self.F = 0.5  # DE mutation factor
        self.CR = 0.9  # DE crossover rate
        self.c1 = 1.5  # PSO cognitive component
        self.c2 = 1.5  # PSO social component
        self.w = 0.5   # PSO inertia weight

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.full(self.population_size, np.inf)

        for _ in range(self.population_size):  # Initial evaluation
            for i, candidate in enumerate(population):
                score = func(candidate)
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = candidate

                if score < self.f_opt:
                    self.f_opt = score
                    self.x_opt = candidate

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Differential Evolution (DE) Mutation
                candidates = list(range(self.population_size))
                candidates.remove(i)
                a, b, c = np.random.choice(candidates, 3, replace=False)
                mutant = population[a] + self.F * (population[b] - population[c])
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Crossover
                trial = np.copy(population[i])
                for j in range(self.dim):
                    if np.random.rand() < self.CR:
                        trial[j] = mutant[j]

                # PSO update of velocities and positions
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best_positions[i] - population[i]) +
                                 self.c2 * r2 * (self.x_opt - population[i]))
                population[i] = np.clip(population[i] + velocities[i], func.bounds.lb, func.bounds.ub)

                # Evaluate trial vector
                score = func(trial)
                evaluations += 1

                # Selection
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = trial

                if score < self.f_opt:
                    self.f_opt = score
                    self.x_opt = trial

                if evaluations >= self.budget:
                    break

        return self.f_opt, self.x_opt