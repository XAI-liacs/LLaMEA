import numpy as np

class HybridGA_PSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50  # Size of population
        self.c1 = 1.5  # Cognitive constant
        self.c2 = 1.5  # Social constant
        self.inertia = 0.5  # Inertia weight
        self.mutation_rate = 0.1  # Mutation rate for GA
        self.mutation_strength = 0.1  # Mutation strength
        self.velocity = np.zeros((self.population_size, dim))
        self.history = []  # Keep track of best solutions

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(x) for x in pop])
        best_global = pop[np.argmin(fitness)]
        best_personal = pop.copy()
        best_personal_fitness = fitness.copy()

        evaluations = self.population_size

        while evaluations < self.budget:
            # Update velocities and positions (PSO Step)
            r1, r2 = np.random.rand(2)
            self.velocity = (self.inertia * self.velocity +
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

            # Apply genetic algorithm operators
            for i in range(self.population_size):
                if np.random.rand() < self.mutation_rate:
                    pop[i] += np.random.normal(0, self.mutation_strength, self.dim)
                    pop[i] = np.clip(pop[i], lb, ub)

            # Save the history of best solutions
            self.history.append(best_global)

        return best_global