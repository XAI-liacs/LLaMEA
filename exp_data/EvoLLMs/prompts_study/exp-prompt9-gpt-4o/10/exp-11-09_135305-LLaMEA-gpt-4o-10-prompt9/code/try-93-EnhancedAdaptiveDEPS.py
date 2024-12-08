import numpy as np

class EnhancedAdaptiveDEPS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20 + int(3.0 * np.sqrt(self.dim))
        self.global_best = None
        self.best_cost = float('inf')
        self.init_population_size = self.population_size
        self.velocities = np.zeros((self.population_size, self.dim))  # Initialize velocities for PSO component

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evals = self.population_size
        F = 0.8
        CR = 0.9
        c1 = 1.5  # Cognitive component
        c2 = 1.5  # Social component
        inertia_weight = 0.5 + 0.4 * np.random.rand()

        while evals < self.budget:
            # Adjust population size dynamically
            self.population_size = self.init_population_size - int(evals / self.budget * (self.init_population_size - 5))

            for i in range(self.population_size):
                # Select three random indices different from i
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                # Perform mutation (differential vector)
                mutant = np.clip(a + F * (b - c), self.lower_bound, self.upper_bound)

                # Perform crossover
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, population[i])

                # Evaluate the trial solution
                trial_cost = func(trial)
                evals += 1

                # Selection
                if trial_cost < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_cost

                    # Update global best
                    if trial_cost < self.best_cost:
                        self.global_best = trial
                        self.best_cost = trial_cost

                # PSO component: update velocities and positions
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                self.velocities[i] = (inertia_weight * self.velocities[i] +
                                      c1 * r1 * (population[i] - trial) +
                                      c2 * r2 * (self.global_best - population[i]))
                population[i] = np.clip(population[i] + self.velocities[i], self.lower_bound, self.upper_bound)

                if evals >= self.budget:
                    break

            # Adaptive F, CR, and inertia weight
            F = 0.5 + 0.3 * np.random.rand()
            CR = 0.8 + 0.1 * np.random.rand()
            inertia_weight = 0.5 + 0.4 * np.random.rand()

        return self.global_best