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
        cooling_rate = 0.95
        swap_interval = 10  # New: interval for swapping individuals
        num_subpops = 5  # New: number of subpopulations

        # Initialize population
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        subpop_size = pop_size // num_subpops  # New: size of each subpopulation
        populations = [np.random.uniform(lb, ub, (subpop_size, self.dim)) for _ in range(num_subpops)]
        pop_fitness = [np.array([func(ind) for ind in pop]) for pop in populations]

        # Initialize best solution
        best_solution = None
        best_fitness = np.inf

        evaluations = pop_size

        while evaluations < self.budget and temp_initial > temp_min:
            for p_index, pop in enumerate(populations):  # New: iterate over subpopulations
                for i in range(subpop_size):
                    if evaluations >= self.budget:
                        break

                    # Adaptive Differential Evolution Mutation
                    idxs = [idx for idx in range(subpop_size) if idx != i]
                    a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                    F_adaptive = 0.5 + 0.5 * (1 - evaluations / self.budget) * np.random.rand()
                    mutant = np.clip(a + F_adaptive * (b - c), lb, ub)

                    # Crossover
                    CR_adaptive = 0.7 + 0.3 * np.random.rand()
                    crossover_mask = np.random.rand(self.dim) < CR_adaptive
                    trial = np.where(crossover_mask, mutant, pop[i])

                    # Evaluate trial vector
                    trial_fitness = func(trial)
                    evaluations += 1

                    # Selection based on Simulated Annealing acceptance criteria
                    if (trial_fitness < pop_fitness[p_index][i]) or (np.random.rand() < np.exp((pop_fitness[p_index][i] - trial_fitness) / temp_initial)):
                        pop[i] = trial
                        pop_fitness[p_index][i] = trial_fitness

                        # Update global best
                        if trial_fitness < best_fitness:
                            best_solution = trial
                            best_fitness = trial_fitness

            # New: Swap individuals between subpopulations
            if evaluations % swap_interval == 0:
                for j in range(num_subpops - 1):
                    swap_idx = np.random.choice(subpop_size, 2, replace=False)
                    populations[j][swap_idx[0]], populations[j+1][swap_idx[1]] = populations[j+1][swap_idx[1]], populations[j][swap_idx[0]]
                    pop_fitness[j][swap_idx[0]], pop_fitness[j+1][swap_idx[1]] = pop_fitness[j+1][swap_idx[1]], pop_fitness[j][swap_idx[0]]

            # Simulated Annealing temperature update
            temp_initial *= cooling_rate * (1 - evaluations / self.budget)

        return best_solution, best_fitness