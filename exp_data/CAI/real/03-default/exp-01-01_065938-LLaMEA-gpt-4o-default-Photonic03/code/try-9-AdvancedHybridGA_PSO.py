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
        self.mutation_rate = 0.1
        self.mutation_strength = 0.1
        self.velocity = np.zeros((self.population_size, dim))
        self.history = []
        self.chaos_factor = 0.5  # Initial value for chaotic sequence

    def chaotic_map(self):
        # Simple logistic map for chaotic sequence generation
        self.chaos_factor = 4 * self.chaos_factor * (1 - self.chaos_factor)
        return self.chaos_factor

    def levy_flight(self, L, d):
        # Lévy flight step size calculation
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, d)
        v = np.random.normal(0, 1, d)
        step = u / np.abs(v)**(1 / beta)
        return L * step

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
            inertia = (self.inertia_max - ((self.inertia_max - self.inertia_min) * 
                      (evaluations / self.budget)) * self.chaotic_map())

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

            avg_fitness = np.mean(fitness)
            if np.min(fitness) < avg_fitness:
                self.mutation_rate = max(0.01, self.mutation_rate * 0.9)
                self.mutation_strength *= 0.9
            else:
                self.mutation_rate = min(0.2, self.mutation_rate * 1.1)
                self.mutation_strength *= 1.1

            for i in range(self.population_size):
                if np.random.rand() < self.mutation_rate:
                    pop[i] += np.random.normal(0, self.mutation_strength, self.dim)
                    pop[i] = np.clip(pop[i], lb, ub)

                if np.random.rand() < 0.1:
                    L = 0.01  # Lévy flight scaling factor
                    pop[i] += self.levy_flight(L, self.dim)
                    pop[i] = np.clip(pop[i], lb, ub)

            self.history.append(best_global)

        return best_global