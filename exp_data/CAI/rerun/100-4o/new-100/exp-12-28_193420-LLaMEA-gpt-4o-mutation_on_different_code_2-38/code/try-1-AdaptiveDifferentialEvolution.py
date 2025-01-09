import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=50):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize population within bounds
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]

        # Differential Evolution parameters
        F = 0.8  # Mutation factor
        CR = 0.9  # Crossover probability
        evaluations = len(fitness)
        success_rate = 0.0  # Track success rate of generations

        while evaluations < self.budget:
            successful_mutations = 0
            for i in range(self.pop_size):
                # Select three random indices different from i
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                # Create mutant vector
                mutant = np.clip(a + F * (b - c), func.bounds.lb, func.bounds.ub)

                # Crossover
                trial = np.copy(population[i])
                crossover_points = np.random.rand(self.dim) < CR
                trial[crossover_points] = mutant[crossover_points]

                # Selection
                f_trial = func(trial)
                evaluations += 1

                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    successful_mutations += 1

                    # Update best solution found
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

                # Break if budget is exceeded
                if evaluations >= self.budget:
                    break

            # Adapt F based on success rate
            success_rate = successful_mutations / self.pop_size
            F = 0.5 + 0.3 * success_rate  # Adaptive modification of F

        return self.f_opt, self.x_opt