import numpy as np

class EMOOAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.max_generations = budget // self.population_size
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        def random_population():
            return np.random.uniform(self.lower_bound, self.upper_bound, size=(self.population_size, self.dim))

        def differential_evolution(population, f=0.5, cr=0.9):
            mutated = population + f * (population[np.random.choice(self.population_size, size=self.population_size, replace=True)] - population)
            crossover = np.random.rand(self.population_size, self.dim) < cr
            trial_population = np.where(crossover, mutated, population)
            return trial_population

        def particle_swarm_optimization(population, w=0.5, c1=1.5, c2=1.5):
            velocity = np.zeros((self.population_size, self.dim))
            pbest = population.copy()
            pbest_fitness = np.array([func(ind) for ind in pbest])
            gbest_idx = np.argmin(pbest_fitness)
            gbest = pbest[gbest_idx]

            for _ in range(self.max_generations):
                r1, r2 = np.random.rand(2, self.population_size, self.dim)
                velocity = w * velocity + c1 * r1 * (pbest - population) + c2 * r2 * (gbest - population)
                population = population + velocity
                fitness_values = np.array([func(ind) for ind in population])

                improved = fitness_values < pbest_fitness
                pbest[improved] = population[improved]
                pbest_fitness[improved] = fitness_values[improved]

                gbest_idx = np.argmin(pbest_fitness)
                gbest = pbest[gbest_idx]

            return gbest

        current_population = random_population()
        best_solution = particle_swarm_optimization(current_population)
        return best_solution