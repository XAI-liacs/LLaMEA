import numpy as np

class EnhancedHybridDE_PSO_GA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 1.5
        self.c2 = 1.5
        self.inertia_max = 0.9
        self.inertia_min = 0.4
        self.mutation_rate = 0.1
        self.mutation_strength = 0.1
        self.velocity = np.zeros((self.population_size, dim))
        self.history = []

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(x) for x in pop])
        best_global = pop[np.argmin(fitness)]
        best_personal = pop.copy()
        best_personal_fitness = fitness.copy()

        evaluations = self.population_size
        inertia = self.inertia_max
        
        F = 0.5  # Differential Evolution mutation factor
        CR = 0.7  # Crossover probability

        while evaluations < self.budget:
            inertia = self.inertia_max - ((self.inertia_max - self.inertia_min) * (evaluations / self.budget))
            
            # Differential Evolution step
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = pop[indices]
                mutant = np.clip(x1 + F * (x2 - x3), lb, ub)
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, pop[i])
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < func(best_global):
                        best_global = trial

            # PSO Step
            r1, r2 = np.random.rand(2)
            self.velocity = (inertia * self.velocity +
                             self.c1 * r1 * (best_personal - pop) +
                             self.c2 * r2 * (best_global - pop))
            pop += self.velocity
            pop = np.clip(pop, lb, ub)

            # Evaluate the new population
            new_fitness = np.array([func(x) for x in pop])
            evaluations += self.population_size

            # Update personal and global best
            update_mask = new_fitness < best_personal_fitness
            best_personal[update_mask] = pop[update_mask]
            best_personal_fitness[update_mask] = new_fitness[update_mask]

            if np.min(new_fitness) < func(best_global):
                best_global = pop[np.argmin(new_fitness)]

            # Adaptive mutation
            avg_fitness = np.mean(new_fitness)
            if np.min(new_fitness) < avg_fitness:
                self.mutation_rate = max(0.01, self.mutation_rate * 0.9)
                self.mutation_strength *= 0.9
            else:
                self.mutation_rate = min(0.2, self.mutation_rate * 1.1)
                self.mutation_strength *= 1.1

            # Apply genetic algorithm operators
            for i in range(self.population_size):
                if np.random.rand() < self.mutation_rate:
                    pop[i] += np.random.normal(0, self.mutation_strength, self.dim)
                    pop[i] = np.clip(pop[i], lb, ub)

            # Save the history of best solutions
            self.history.append(best_global)

        return best_global