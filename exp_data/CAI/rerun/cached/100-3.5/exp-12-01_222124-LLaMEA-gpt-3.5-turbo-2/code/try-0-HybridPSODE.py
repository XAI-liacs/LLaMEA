import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim, pop_size=30, w=0.5, c1=1.494, c2=1.494, f=0.5):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.f = f

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))

        def pso_update_position(pop, vel):
            return pop + vel

        def de_update_position(pop, best, f):
            mutant_pop = best + f * (pop - best)
            return np.clip(mutant_pop, -5.0, 5.0)

        def evaluate_population(population):
            return np.array([func(individual) for individual in population])

        population = initialize_population()
        velocity = np.zeros((self.pop_size, self.dim))

        for _ in range(self.budget):
            fitness = evaluate_population(population)

            best_idx = np.argmin(fitness)
            best_individual = population[best_idx]

            for i in range(self.pop_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                
                # PSO update
                velocity[i] = self.w * velocity[i] + self.c1 * r1 * (best_individual - population[i]) + self.c2 * r2 * (best_individual - population[i])
                population = pso_update_position(population, velocity)

                # DE update
                mutant = de_update_position(population[i], best_individual, self.f)
                population[i] = mutant if func(mutant) < func(population[i]) else population[i]

        best_idx = np.argmin(evaluate_population(population))
        best_solution = population[best_idx]
        return best_solution