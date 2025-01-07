import numpy as np

class EnhancedHybridGA_PSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 1.5  # Cognitive constant
        self.c2 = 1.5  # Social constant
        self.inertia = 0.9  # Initial inertia weight 
        self.final_inertia = 0.4  # Final inertia weight
        self.mutation_rate = 0.1
        self.mutation_strength = 0.1
        self.velocity = np.zeros((self.population_size, dim))
        self.history = []
        self.generations = self.budget // self.population_size

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(x) for x in pop])
        best_global = pop[np.argmin(fitness)]
        best_personal = pop.copy()
        best_personal_fitness = fitness.copy()

        evaluations = self.population_size

        for gen in range(self.generations):
            # Dynamically update inertia weight
            inertia = self.final_inertia + (self.inertia - self.final_inertia) * ((self.generations - gen) / self.generations)

            # Update velocities and positions (PSO Step)
            r1, r2 = np.random.rand(2)
            self.velocity = (inertia * self.velocity +
                             self.c1 * r1 * (best_personal - pop) +
                             self.c2 * r2 * (best_global - pop))
            pop += self.velocity
            pop = np.clip(pop, lb, ub)

            # Evaluate the new population
            fitness = np.array([func(x) for x in pop])
            evaluations += self.population_size

            # Update personal and global best
            update_mask = fitness < best_personal_fitness
            best_personal[update_mask] = pop[update_mask]
            best_personal_fitness[update_mask] = fitness[update_mask]

            if np.min(fitness) < func(best_global):
                best_global = pop[np.argmin(fitness)]

            # Tournament selection for genetic algorithm
            for i in range(self.population_size):
                if np.random.rand() < self.mutation_rate:
                    candidates = np.random.choice(self.population_size, size=3, replace=False)
                    winner = candidates[np.argmin(fitness[candidates])]
                    pop[i] = pop[winner] + np.random.normal(0, self.mutation_strength, self.dim)
                    pop[i] = np.clip(pop[i], lb, ub)

            # Save the history of best solutions
            self.history.append(best_global)

        return best_global