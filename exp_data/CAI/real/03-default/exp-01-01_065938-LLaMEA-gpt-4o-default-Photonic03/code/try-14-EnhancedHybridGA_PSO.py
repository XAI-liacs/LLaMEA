import numpy as np

class EnhancedHybridGA_PSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 1.5  # Cognitive constant
        self.c2 = 1.5  # Social constant
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
        neighborhood_size = 5  # Consider a small neighborhood

        while evaluations < self.budget:
            inertia = self.inertia_max - ((self.inertia_max - self.inertia_min) * (evaluations / self.budget))
            r1, r2 = np.random.rand(2)

            # Update velocities and positions (PSO Step)
            for i in range(self.population_size):
                # Select neighborhood
                neighbors = np.random.choice(self.population_size, size=neighborhood_size, replace=False)
                best_neighbor = pop[neighbors[np.argmin(fitness[neighbors])]]

                # Neighborhood influence alongside global influence
                self.velocity[i] = (inertia * self.velocity[i] +
                                    self.c1 * r1 * (best_personal[i] - pop[i]) +
                                    self.c2 * r2 * (best_global - pop[i]) +
                                    self.c2 * r2 * (best_neighbor - pop[i]) * 0.5)

            pop += self.velocity
            pop = np.clip(pop, lb, ub)

            fitness = np.array([func(x) for x in pop])
            evaluations += self.population_size

            # Update personal and global best
            update_mask = fitness < best_personal_fitness
            best_personal[update_mask] = pop[update_mask]
            best_personal_fitness[update_mask] = fitness[update_mask]

            if np.min(fitness) < func(best_global):
                best_global = pop[np.argmin(fitness)]

            avg_fitness = np.mean(fitness)
            if np.min(fitness) < avg_fitness:
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

            self.history.append(best_global)

        return best_global