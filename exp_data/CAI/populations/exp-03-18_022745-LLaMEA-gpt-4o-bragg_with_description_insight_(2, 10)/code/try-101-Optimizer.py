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

        def periodicity_penalty(solution, gen, max_gen):
            period = 3  
            penalty_weight = 1 + 0.8 * (gen / max_gen)  # Adjusted dynamic weight
            penalty = penalty_weight * np.sum((solution[0::2] - solution[1::2]) ** 4)
            return penalty
        
        population = np.random.uniform(lb, ub, (pop_size, self.dim))
        opposite_pop = opposite_population(population, lb, ub)
        population = np.vstack((population, opposite_pop))
        function_values = np.apply_along_axis(func, 1, population) + periodicity_penalty(population, 0, self.budget // (2 * pop_size))

        memory_F, memory_CR = [], []  # New memory for F and CR
        for gen in range(self.budget // (2 * pop_size)):
            F_dynamic = 0.5 + 0.4 * np.sin(np.pi * gen / 15)
            CR_dynamic = 0.7 + 0.2 * np.cos(np.pi * gen / 10)  # Enhanced adaptive CR with different frequency
            if len(memory_F) > 0:  # Use memory if available
                F_dynamic = np.mean(memory_F)
                CR_dynamic = np.mean(memory_CR)
            for i in range(population.shape[0]):
                indices = np.random.choice(population.shape[0], 3, replace=False)
                x_r1, x_r2, x_r3 = population[indices]
                mutant_vector = x_r1 + F_dynamic * (x_r2 - x_r3)
                trial_vector = np.where(np.random.rand(self.dim) < CR_dynamic, mutant_vector, population[i])
                trial_vector = np.clip(trial_vector, lb, ub)

                trial_value = func(trial_vector) + periodicity_penalty(trial_vector, gen, self.budget // (2 * pop_size))
                if trial_value < function_values[i]:
                    population[i] = trial_vector
                    function_values[i] = trial_value
                    memory_F.append(F_dynamic)  # Store successful parameters
                    memory_CR.append(CR_dynamic)

        best_idx = np.argmin(function_values)
        return population[best_idx]

    def local_refinement(self, func, x0, lb, ub):
        result = minimize(lambda x: func(x) + 0.15 * np.sum((x[0::2] - x[1::2]) ** 2), x0,
                          bounds=[(low, high) for low, high in zip(lb, ub)], method='L-BFGS-B')
        return result.x

    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        pop_size, F, CR = 20, 0.8, 0.9
        best_solution = self.quasi_oppositional_de(func, lb, ub, pop_size, F, CR)
        refined_solution = self.local_refinement(func, best_solution, lb, ub)
        return refined_solution