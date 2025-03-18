import numpy as np

class PeriodicityEnhancedDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Hyperparameters
        pop_size = 20
        F = 0.8  # Differential weight
        Cr = 0.9  # Crossover probability
        periodicity_weight = 0.1
        local_search_prob = 0.3

        # Initialize population
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (pop_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        
        num_evaluations = pop_size

        while num_evaluations < self.budget:
            for i in range(pop_size):
                # Mutation
                indices = [idx for idx in range(pop_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), lb, ub)
                
                # Crossover
                cross_points = np.random.rand(self.dim) < Cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Evaluate trial solution
                trial_fitness = func(trial)
                num_evaluations += 1

                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

            # Periodicity enforcement
            if periodicity_weight > 0:
                periodicity_weight = periodicity_weight * (1 - num_evaluations/self.budget)  # Dynamic adjustment
                for i in range(pop_size):
                    new_sol = self.enforce_periodicity(population[i], periodicity_weight)
                    new_fitness = func(new_sol)
                    num_evaluations += 1
                    if new_fitness < fitness[i]:
                        population[i] = new_sol
                        fitness[i] = new_fitness

            # Local search
            if np.random.rand() < local_search_prob:
                for i in range(pop_size):
                    new_sol = self.local_search(population[i], func)
                    new_fitness = func(new_sol)
                    num_evaluations += 1
                    if new_fitness < fitness[i]:
                        population[i] = new_sol
                        fitness[i] = new_fitness

            if num_evaluations >= self.budget:
                break

        best_index = np.argmin(fitness)
        return population[best_index]

    def enforce_periodicity(self, solution, weight):
        avg_period = np.mean(solution)
        return solution + weight * (avg_period - solution)

    def local_search(self, solution, func):
        epsilon = 1e-4
        best_sol = solution.copy()
        best_fitness = func(best_sol)
        for i in range(len(solution)):
            perturb = np.zeros_like(solution)
            perturb[i] = epsilon
            trial_sol = np.clip(best_sol + perturb, func.bounds.lb, func.bounds.ub)
            trial_fitness = func(trial_sol)
            if trial_fitness < best_fitness:
                best_sol = trial_sol
                best_fitness = trial_fitness
        return best_sol