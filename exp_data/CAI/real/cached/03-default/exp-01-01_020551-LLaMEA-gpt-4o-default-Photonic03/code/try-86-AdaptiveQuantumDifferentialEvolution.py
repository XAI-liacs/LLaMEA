import numpy as np

class AdaptiveQuantumDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, 5 * dim)  # Scale population size according to the problem dimensionality
        self.f_cr_initial = (0.5, 0.9)  # Initial crossover rate range
        self.f_min, self.f_max = 0.5, 1.0  # Differential weight range
        self.beta = 0.2  # Quantum perturbation factor
        self.adaptive_rate = 0.005  # Rate of adaptation for control parameters

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        scores = np.array([func(individual) for individual in population])
        evaluations = self.population_size
        f_cr = np.random.uniform(*self.f_cr_initial)

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                # Mutation - DE/rand/1
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = a + np.random.uniform(self.f_min, self.f_max) * (b - c)
                mutant = np.clip(mutant, lb, ub)

                # Crossover
                crossover = np.random.rand(self.dim) < f_cr
                trial = np.where(crossover, mutant, population[i])

                # Quantum-inspired perturbation
                if np.random.rand() < self.beta:
                    q_step = np.random.normal(0, 0.5, self.dim)
                    trial += q_step * (ub - lb)

                trial = np.clip(trial, lb, ub)
                trial_score = func(trial)
                evaluations += 1

                # Selection
                if trial_score < scores[i]:
                    population[i] = trial
                    scores[i] = trial_score

            # Update best solution
            best_index = np.argmin(scores)
            best_individual = population[best_index].copy()

            # Adapt control parameters
            f_cr = max(self.f_cr_initial[0], f_cr - self.adaptive_rate)
            self.beta = min(1.0, self.beta + self.adaptive_rate * (1 - evaluations / self.budget))

        return best_individual, scores[best_index]