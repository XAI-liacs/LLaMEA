import numpy as np

class HybridDEAnnealingOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize parameters
        pop_size = min(50, self.budget // 10)  # population size
        F = 0.85  # differential weight
        CR = 0.9  # crossover probability
        temp_initial = 1000.0
        temp_min = 1e-5
        cooling_rate = 0.98  # More aggressive cooling

        # Initialize population
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        pop = np.random.uniform(lb, ub, (pop_size, self.dim))
        pop_fitness = np.array([func(ind) for ind in pop])

        # Initialize best solution
        best_idx = np.argmin(pop_fitness)
        best_solution = pop[best_idx]
        best_fitness = pop_fitness[best_idx]

        evaluations = pop_size

        while evaluations < self.budget and temp_initial > temp_min:
            for i in range(pop_size):
                if evaluations >= self.budget:
                    break

                # Adaptive Differential Evolution Mutation
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                F_adaptive = 0.5 + 0.5 * (1 - evaluations / self.budget) * np.random.rand()  # Introduce randomness in F
                mutant = np.clip(a + F_adaptive * (b - c), lb, ub)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < CR
                trial = np.where(crossover_mask, mutant, pop[i])

                # Evaluate trial vector
                trial_fitness = func(trial)
                evaluations += 1

                # Selection based on Simulated Annealing acceptance criteria
                if (trial_fitness < pop_fitness[i]) or (np.random.rand() < np.exp((pop_fitness[i] - trial_fitness) / temp_initial)):
                    pop[i] = trial
                    pop_fitness[i] = trial_fitness

                    # Update global best
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

            # Simulated Annealing temperature update
            temp_initial *= cooling_rate * (1 - evaluations / self.budget)  # Adaptive cooling

        return best_solution, best_fitness