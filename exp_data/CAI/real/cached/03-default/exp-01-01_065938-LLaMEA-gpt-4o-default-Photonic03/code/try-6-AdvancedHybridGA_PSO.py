import numpy as np

class AdvancedHybridGA_PSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.c1 = 1.5
        self.c2 = 1.5
        self.inertia_max = 0.9
        self.inertia_min = 0.4
        self.velocity = np.zeros((self.population_size, dim))
        self.crossover_rate = 0.7
        self.mutation_rate = 0.1
        self.mutation_strength = 0.1
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

        while evaluations < self.budget:
            inertia = self.inertia_max - ((self.inertia_max - self.inertia_min) * (evaluations / self.budget))
            r1, r2 = np.random.rand(2)
            self.velocity = (inertia * self.velocity +
                             self.c1 * r1 * (best_personal - pop) +
                             self.c2 * r2 * (best_global - pop))
            pop += self.velocity
            pop = np.clip(pop, lb, ub)

            fitness = np.array([func(x) for x in pop])
            evaluations += self.population_size

            update_mask = fitness < best_personal_fitness
            best_personal[update_mask] = pop[update_mask]
            best_personal_fitness[update_mask] = fitness[update_mask]

            if np.min(fitness) < func(best_global):
                best_global = pop[np.argmin(fitness)]

            # Adaptive mutation and crossover based on diversity
            diversity = np.std(pop)
            if diversity < 0.1 * (ub - lb):
                self.crossover_rate = min(0.9, self.crossover_rate * 1.1)
                self.mutation_rate = max(0.01, self.mutation_rate * 0.9)
                self.mutation_strength *= 0.9
            else:
                self.crossover_rate = max(0.5, self.crossover_rate * 0.9)
                self.mutation_rate = min(0.2, self.mutation_rate * 1.1)
                self.mutation_strength *= 1.1

            # Genetic Algorithm: Crossover
            for i in range(0, self.population_size, 2):
                if np.random.rand() < self.crossover_rate:
                    parent1, parent2 = pop[i], pop[i+1]
                    alpha = np.random.rand(self.dim)
                    child1 = alpha * parent1 + (1 - alpha) * parent2
                    child2 = alpha * parent2 + (1 - alpha) * parent1
                    pop[i], pop[i+1] = np.clip(child1, lb, ub), np.clip(child2, lb, ub)

            # Genetic Algorithm: Mutation
            for i in range(self.population_size):
                if np.random.rand() < self.mutation_rate:
                    pop[i] += np.random.normal(0, self.mutation_strength, self.dim)
                    pop[i] = np.clip(pop[i], lb, ub)

            self.history.append(best_global)

        return best_global