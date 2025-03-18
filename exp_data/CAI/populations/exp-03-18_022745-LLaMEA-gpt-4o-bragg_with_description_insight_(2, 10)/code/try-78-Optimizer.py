import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def quasi_oppositional_de(self, func, lb, ub, pop_size, F, CR):
        def opposite_population(pop, lb, ub):
            opp_pop = lb + ub - pop
            return opp_pop

        def periodicity_penalty(solution, gen):
            period = 3
            penalty_weight = 0.1 + 0.9 * (1 - gen / (self.budget // (2 * pop_size)))  # Dynamic weight
            penalty = penalty_weight * np.sum((solution[0::2] - solution[1::2]) ** 4)
            return penalty

        population = np.random.uniform(lb, ub, (pop_size, self.dim))
        opposite_pop = opposite_population(population, lb, ub)
        population = np.vstack((population, opposite_pop))
        function_values = np.apply_along_axis(func, 1, population) + periodicity_penalty(population, 0)

        for gen in range(self.budget // (2 * pop_size)):
            F_dynamic = 0.5 + 0.3 * np.sin(np.pi * gen / 10)  # Adaptive F
            CR_dynamic = 0.8 + 0.1 * np.cos(np.pi * gen / 10)  # Adaptive CR
            for i in range(population.shape[0]):
                indices = np.random.choice(population.shape[0], 3, replace=False)
                x_r1, x_r2, x_r3 = population[indices]
                mutant_vector = x_r1 + F_dynamic * (x_r2 - x_r3)
                trial_vector = np.where(np.random.rand(self.dim) < CR_dynamic, mutant_vector, population[i])
                trial_vector = np.clip(trial_vector, lb, ub)

                trial_value = func(trial_vector) + periodicity_penalty(trial_vector, gen)
                if trial_value < function_values[i]:
                    population[i] = trial_vector
                    function_values[i] = trial_value

        best_idx = np.argmin(function_values)
        return population[best_idx]

    def local_refinement(self, func, x0, lb, ub):
        result = minimize(lambda x: func(x) + 0.1 * np.sum((x[0::2] - x[1::2]) ** 2), x0,  
                          bounds=[(low, high) for low, high in zip(lb, ub)], method='L-BFGS-B')
        return result.x

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        pop_size, F, CR = 20, 0.8, 0.9
        best_solution = self.quasi_oppositional_de(func, lb, ub, pop_size, F, CR)
        refined_solution = self.local_refinement(func, best_solution, lb, ub)
        return refined_solution