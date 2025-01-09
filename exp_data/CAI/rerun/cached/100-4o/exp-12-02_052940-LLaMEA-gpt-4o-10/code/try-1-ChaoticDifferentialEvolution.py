import numpy as np

class ChaoticDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.F = 0.5  # differential weight
        self.CR = 0.9  # crossover probability
        self.chaotic_map = self.logistic_map
        self.num_evaluations = 0

    def logistic_map(self, x):
        return 4.0 * x * (1.0 - x)

    def chaotic_sequence(self, length):
        x = 0.7  # initial value
        seq = []
        for _ in range(length):
            x = self.chaotic_map(x)
            seq.append(x)
        return seq

    def __call__(self, func):
        pop = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        self.num_evaluations += self.population_size

        best_idx = np.argmin(fitness)
        best = pop[best_idx]

        chaotic_seq = self.chaotic_sequence(self.budget)

        while self.num_evaluations < self.budget:
            for i in range(self.population_size):
                if self.num_evaluations >= self.budget:
                    break

                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]

                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)

                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, pop[i])

                f_trial = func(trial)
                self.num_evaluations += 1

                if f_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial

                    if f_trial < fitness[best_idx]:
                        best_idx = i
                        best = trial

                # Adapt parameters using chaotic sequence
                self.F = 0.5 + 0.3 * chaotic_seq[self.num_evaluations % len(chaotic_seq)]
                self.CR = 0.8 + 0.2 * chaotic_seq[self.num_evaluations % len(chaotic_seq)]

        return best