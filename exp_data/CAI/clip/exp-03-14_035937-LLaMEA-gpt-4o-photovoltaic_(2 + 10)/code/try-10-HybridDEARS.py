import numpy as np

class HybridDEARS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        np.random.seed(0)  # For reproducibility

        pop_size = 10 * self.dim
        mutation_factor = 0.8
        crossover_probability = 0.9
        adaptive_threshold = 0.1  # Start with a small exploration

        # Initialize the population within bounds
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        best_idx = np.argmin(fitness)
        best_solution = pop[best_idx]
        best_fitness = fitness[best_idx]
        eval_count = pop_size

        while eval_count < self.budget:
            for i in range(pop_size):
                if eval_count >= self.budget:
                    break

                # Tournament selection for better candidate selection
                candidates = np.random.choice(range(pop_size), 5, replace=False)
                a, b, c = sorted(candidates, key=lambda idx: fitness[idx])[:3]

                # Mutation
                mutant = np.clip(pop[a] + mutation_factor * (pop[b] - pop[c]), func.bounds.lb, func.bounds.ub)

                # Crossover
                cross_points = np.random.rand(self.dim) < crossover_probability
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # Fitness evaluation
                trial_fitness = func(trial)
                eval_count += 1

                # Selection
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

            # Adaptive Random Search
            if eval_count < self.budget and np.random.rand() < adaptive_threshold:
                for i in range(pop_size):
                    if eval_count >= self.budget:
                        break
                    random_solution = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
                    random_fitness = func(random_solution)
                    eval_count += 1

                    if random_fitness < fitness[i]:
                        pop[i] = random_solution
                        fitness[i] = random_fitness
                        if random_fitness < best_fitness:
                            best_solution = random_solution
                            best_fitness = random_fitness

            # Adapt exploration rate over time
            adaptive_threshold = max(0.01, adaptive_threshold * 0.95)

        return best_solution, best_fitness