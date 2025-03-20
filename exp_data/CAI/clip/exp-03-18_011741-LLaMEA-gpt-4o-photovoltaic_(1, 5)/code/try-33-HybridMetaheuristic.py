import numpy as np

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        pop_size = 10 * self.dim
        F_base = 0.5
        CR_base = 0.9
        
        # Initialize parameters for Particle Swarm Optimization (PSO)
        c1 = 1.5
        c2 = 1.5
        w = 0.9 - (0.9 - 0.4) * (self.budget / pop_size)
        
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (pop_size, self.dim))
        velocity = np.random.uniform(-1, 1, (pop_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        
        p_best = np.copy(population)
        p_best_fitness = np.copy(fitness)
        g_best = population[np.argmin(fitness)]
        g_best_fitness = np.min(fitness)

        evaluations = pop_size

        chaotic_sequence = np.random.rand(pop_size)

        while evaluations < self.budget:
            for i in range(pop_size):
                # Differential Evolution Mutation and Crossover
                indices = list(range(pop_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                adaptive_rate = np.abs(np.sin(np.pi * evaluations / self.budget))

                F = F_base * adaptive_rate
                CR = CR_base * adaptive_rate

                mutant = population[a] + F * (population[b] - population[c])
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
                trial = np.copy(population[i])
                
                crossover_points = np.random.rand(self.dim) < CR
                trial[crossover_points] = mutant[crossover_points]
                
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < p_best_fitness[i]:
                        p_best[i] = trial
                        p_best_fitness[i] = trial_fitness

                if trial_fitness < g_best_fitness:
                    g_best = trial
                    g_best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break
            
            c1 = 2.0 - (2.0 - 0.5) * (evaluations / self.budget)
            c2 = 2.5 - (2.5 - 1.0) * (evaluations / self.budget)
            
            r1, r2 = np.random.rand(2)
            velocity = w * velocity + c1 * r1 * (p_best - population) + c2 * r2 * (g_best - population)
            population = population + velocity
            population = np.clip(population, func.bounds.lb, func.bounds.ub)
            
            chaotic_sequence = 4.0 * chaotic_sequence * (1.0 - chaotic_sequence)
            for i in range(pop_size):
                step_size = 0.1 / (1 + 0.01 * evaluations)
                delta = np.random.uniform(-step_size, step_size, self.dim)
                new_solution = np.clip(population[i] + delta * chaotic_sequence[i], func.bounds.lb, func.bounds.ub)
                new_fitness = func(new_solution)
                evaluations += 1

                if new_fitness < fitness[i]:
                    population[i] = new_solution
                    fitness[i] = new_fitness
                    gradient = (new_fitness - fitness[i]) / (np.linalg.norm(delta) + 1e-8)
                    refined_solution = np.clip(new_solution - 0.01 * gradient, func.bounds.lb, func.bounds.ub)
                    refined_fitness = func(refined_solution)
                    if refined_fitness < fitness[i]:
                        population[i] = refined_solution
                        fitness[i] = refined_fitness
                        if refined_fitness < p_best_fitness[i]:
                            p_best[i] = refined_solution
                            p_best_fitness[i] = refined_fitness

        return g_best