import numpy as np

class HybridDEAnnealingOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize parameters
        pop_size = min(50, self.budget // 10)
        F_base = 0.85
        CR = 0.9
        temp_initial = 1000.0
        temp_min = 1e-5
        cooling_rate = 0.95

        # Initialize population
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        pop = np.random.uniform(lb, ub, (pop_size, self.dim))
        pop_fitness = np.array([func(ind) for ind in pop])

        # Initialize best solution
        best_idx = np.argmin(pop_fitness)
        best_solution = pop[best_idx]
        best_fitness = pop_fitness[best_idx]

        evaluations = pop_size
        reinit_counter = 0

        while evaluations < self.budget and temp_initial > temp_min:
            for i in range(pop_size):
                if evaluations >= self.budget:
                    break

                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                F_adaptive = F_base * (0.5 + 0.5 * np.random.rand())
                mutant = np.clip(a + F_adaptive * (b - c), lb, ub)

                crossover_mask = np.random.rand(self.dim) < CR
                trial = np.where(crossover_mask, mutant, pop[i])

                trial_fitness = func(trial)
                evaluations += 1

                if (trial_fitness < pop_fitness[i]) or (np.random.rand() < np.exp((pop_fitness[i] - trial_fitness) / temp_initial)):
                    pop[i] = trial
                    pop_fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

            temp_initial *= cooling_rate * (1 - evaluations / self.budget)

            reinit_counter += 1
            if reinit_counter % 10 == 0:  # Periodic reinitialization
                new_points = np.random.uniform(lb, ub, (pop_size // 5, self.dim))
                for point in new_points:
                    if evaluations >= self.budget:
                        break
                    fitness = func(point)
                    evaluations += 1
                    worst_idx = np.argmax(pop_fitness)
                    if fitness < pop_fitness[worst_idx]:
                        pop[worst_idx] = point
                        pop_fitness[worst_idx] = fitness
                        if fitness < best_fitness:
                            best_solution = point
                            best_fitness = fitness

        return best_solution, best_fitness