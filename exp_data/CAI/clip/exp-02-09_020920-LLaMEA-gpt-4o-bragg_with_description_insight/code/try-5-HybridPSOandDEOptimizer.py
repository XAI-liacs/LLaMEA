import numpy as np
from scipy.optimize import minimize

class HybridPSOandDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population_size = 20
        w = 0.5  # Inertia weight for PSO
        c1 = 1.5 # Cognitive parameter
        c2 = 1.5 # Social parameter
        F = 0.8  # DE scaling factor
        CR = 0.9 # DE crossover rate
        lb, ub = func.bounds.lb, func.bounds.ub

        # Initialize particles
        population = np.random.uniform(lb, ub, (population_size, self.dim))
        velocities = np.random.uniform(-1, 1, population.shape) * (ub - lb)
        personal_best_positions = np.copy(population)
        personal_best_scores = np.array([func(ind) for ind in personal_best_positions])
        
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        evaluations = population_size
        gen = 0  # Generation counter for dynamic adjustments

        while evaluations < self.budget:
            if evaluations + population_size > self.budget:
                break
            
            # Update velocities and positions (PSO component)
            r1, r2 = np.random.rand(population_size, self.dim), np.random.rand(population_size, self.dim)
            velocities = (
                w * velocities
                + c1 * r1 * (personal_best_positions - population)
                + c2 * r2 * (global_best_position - population)
            )
            population = np.clip(population + velocities, lb, ub)

            # DE mutation and crossover
            new_population = np.empty_like(population)
            for i in range(population_size):
                indices = np.arange(population_size)
                indices = indices[indices != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                
                mutant = np.clip(a + F * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                trial = self.dynamic_periodicity(trial, gen)
                
                f_trial = func(trial)
                evaluations += 1

                if f_trial < personal_best_scores[i]:
                    personal_best_positions[i] = trial
                    personal_best_scores[i] = f_trial

                new_population[i] = trial if f_trial < func(population[i]) else population[i]

                if f_trial < global_best_score:
                    global_best_score = f_trial
                    global_best_position = trial

            population = new_population
            gen += 1

            if evaluations < self.budget:
                opt_result = minimize(func, global_best_position, method='L-BFGS-B', bounds=list(zip(lb, ub)))
                evaluations += opt_result.nfev
                if opt_result.fun < global_best_score:
                    global_best_score = opt_result.fun
                    global_best_position = opt_result.x

        return global_best_position

    def dynamic_periodicity(self, solution, generation):
        period = max(2, self.dim // (2 + generation % 3))
        for i in range(0, self.dim, period):
            solution[i:i + period] = np.mean(solution[i:i + period])  # Enforce periodic blocks
        return solution