import numpy as np

class SwarmDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim  # Empirical choice for population size
        self.c1 = 1.5  # Cognitive coefficient
        self.c2 = 1.5  # Social coefficient
        self.f = 0.8   # Differential weight
        self.cr = 0.9  # Crossover probability

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best = population.copy()
        personal_best_fitness = np.array([func(ind) for ind in population])
        global_best = personal_best[np.argmin(personal_best_fitness)]
        global_best_fitness = np.min(personal_best_fitness)

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Update velocity and position (PSO component)
                r1, r2 = np.random.rand(2)
                velocities[i] = velocities[i] + self.c1 * r1 * (personal_best[i] - population[i]) + self.c2 * r2 * (global_best - population[i])
                candidate_position = population[i] + velocities[i]

                # Differential Evolution mutation and crossover
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = population[a] + self.f * (population[b] - population[c])
                trial = np.where(np.random.rand(self.dim) < self.cr, mutant, candidate_position)
                trial = np.clip(trial, self.lower_bound, self.upper_bound)

                # Evaluate new candidate
                trial_fitness = func(trial)
                evaluations += 1

                # Selection and update personal best
                if trial_fitness < personal_best_fitness[i]:
                    personal_best[i] = trial
                    personal_best_fitness[i] = trial_fitness

                # Update global best
                if trial_fitness < global_best_fitness:
                    global_best = trial
                    global_best_fitness = trial_fitness

                # Update the population
                population[i] = trial

                # Break if budget is reached
                if evaluations >= self.budget:
                    break

        return global_best