import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=20):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.f = 0.5  # initial mutation factor
        self.cr = 0.9  # initial crossover probability
        self.history = []  # store past successful mutations

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        fitness = np.array([func(x) for x in population])
        eval_count = self.pop_size

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        generation = 0
        while eval_count < self.budget:
            if generation % 10 == 0 and self.pop_size > 5:
                self.pop_size = max(5, int(self.pop_size * 0.9))  # reduce population size every 10 generations
                population = population[:self.pop_size]
                fitness = fitness[:self.pop_size]

            new_population = np.copy(population)
            for i in range(self.pop_size):
                if eval_count >= self.budget:
                    break

                # Mutation: Select three random individuals
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]

                # Create mutant vector
                mutant_vector = a + self.f * (b - c)
                mutant_vector = np.clip(mutant_vector, func.bounds.lb, func.bounds.ub)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial_vector = np.where(cross_points, mutant_vector, population[i])

                # Selection
                trial_fitness = func(trial_vector)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    new_population[i] = trial_vector
                    fitness[i] = trial_fitness
                    self.history.append((self.f, self.cr))  # record successful rates

                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial_vector

            population = new_population
            generation += 1

            # Adaptive Strategies
            if self.history:
                self.f = np.mean([h[0] for h in self.history]) + 0.1 * np.random.rand()
                self.cr = np.mean([h[1] for h in self.history]) + 0.1 * np.random.rand()

        return self.f_opt, self.x_opt