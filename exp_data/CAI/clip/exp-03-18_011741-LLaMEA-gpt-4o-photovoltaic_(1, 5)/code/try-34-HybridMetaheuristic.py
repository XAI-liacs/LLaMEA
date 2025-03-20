import numpy as np

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Initialize parameters for Differential Evolution (DE)
        pop_size = 10 * self.dim
        F = np.random.uniform(0.5, 0.9)  # Adaptive differential weight
        CR = np.random.uniform(0.5, 0.9)  # Adaptive crossover probability
        
        # Initialize parameters for Particle Swarm Optimization (PSO)
        c1 = 1.5  # Cognitive component
        c2 = 1.5  # Social component
        w = 0.9 - (0.5 - 0.4) * (self.budget / pop_size)  # Refined dynamic inertia weight adjustment
        
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (pop_size, self.dim))
        velocity = np.random.uniform(-1, 1, (pop_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)
        
        # Initialize personal best and global best
        p_best = np.copy(population)
        p_best_fitness = np.copy(fitness)
        g_best = population[np.argmin(fitness)]
        g_best_fitness = np.min(fitness)

        evaluations = pop_size

        while evaluations < self.budget:
            for i in range(pop_size):
                # Differential Evolution Mutation and Crossover
                indices = list(range(pop_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = population[a] + F * (population[b] - population[c])
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)
                trial = np.copy(population[i])
                crossover_points = np.random.rand(self.dim) < CR
                trial[crossover_points] = mutant[crossover_points]
                
                # Evaluate trial solution
                trial_fitness = func(trial)
                evaluations += 1

                # Select between trial and original
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < p_best_fitness[i]:
                        p_best[i] = trial
                        p_best_fitness[i] = trial_fitness

                # Update global best
                if trial_fitness < g_best_fitness:
                    g_best = trial
                    g_best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break
            
            # Adaptive component tuning
            c1 = 2.0 - (2.0 - 0.5) * (evaluations / self.budget)
            c2 = 2.5 - (2.5 - 1.0) * (evaluations / self.budget)
            
            # Particle Swarm Update
            r1, r2 = np.random.rand(2)
            velocity = w * velocity + c1 * r1 * (p_best - population) + c2 * r2 * (g_best - population)
            population = population + velocity
            population = np.clip(population, func.bounds.lb, func.bounds.ub)
            
            # Local search step to refine solutions
            for i in range(pop_size):
                step_size = 0.1 / (1 + 0.01 * evaluations)  # Adaptive step size
                delta = np.random.uniform(-step_size, step_size, self.dim)
                new_solution = np.clip(population[i] + delta, func.bounds.lb, func.bounds.ub)
                new_fitness = func(new_solution)
                evaluations += 1

                if new_fitness < fitness[i]:
                    population[i] = new_solution
                    fitness[i] = new_fitness
                    gradient = (new_fitness - fitness[i]) / (np.linalg.norm(delta) + 1e-8)  # Gradient approximation
                    refined_solution = np.clip(new_solution - 0.01 * gradient, func.bounds.lb, func.bounds.ub)
                    refined_fitness = func(refined_solution)
                    if refined_fitness < fitness[i]:
                        population[i] = refined_solution
                        fitness[i] = refined_fitness
                        if refined_fitness < p_best_fitness[i]:
                            p_best[i] = refined_solution
                            p_best_fitness[i] = refined_fitness

        return g_best