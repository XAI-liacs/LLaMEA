import numpy as np

class HybridDEAnnealingOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize parameters
        pop_size = min(60, self.budget // 10)  # Adjusted population size
        F_initial = 0.9  # initial differential weight
        CR_initial = 0.8  # initial crossover probability
        temp_initial = 1000.0
        temp_min = 1e-5
        cooling_rate = 0.9  # Slightly adjusted cooling rate

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
            F = F_initial * (1 - evaluations / self.budget)  # Adaptive F
            CR = CR_initial * (1 - evaluations / self.budget)  # Adaptive CR

            for i in range(pop_size):
                if evaluations >= self.budget:
                    break

                # Adaptive DE Mutation
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), lb, ub)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < CR
                trial = np.where(crossover_mask, mutant, pop[i])

                # Evaluate trial vector
                trial_fitness = func(trial)
                evaluations += 1

                # Selection based on modified acceptance criteria
                acceptance_probability = np.exp((pop_fitness[i] - trial_fitness) / temp_initial)
                if (trial_fitness < pop_fitness[i]) or (np.random.rand() < acceptance_probability):
                    pop[i] = trial
                    pop_fitness[i] = trial_fitness

                    # Update global best
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

            # Boltzmann-inspired temperature update
            temp_initial *= cooling_rate / (1 + np.log(1 + evaluations))  # Adaptive cooling

        return best_solution, best_fitness