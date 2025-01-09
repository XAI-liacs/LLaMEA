import numpy as np

class DE_PSO_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.max_iter = budget // self.population_size
        self.lb = -5.0
        self.ub = 5.0
        self.c1 = 2.0
        self.c2 = 2.0
        self.w = 0.7
        self.cr = 0.9
        self.f = 0.5

    def __call__(self, func):
        def initialize_population():
            return np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))

        def clipToBounds(population):
            return np.clip(population, self.lb, self.ub)

        def evaluate_population(population):
            return np.array([func(individual) for individual in population])

        def mutate(current, pbest, gbest):
            return current + self.f * (pbest - current) + self.f * (gbest - current)

        def DE_Operator(population, fitness_values):
            pbest_idx = np.argmin(fitness_values)
            gbest_idx = np.argmin(fitness_values)
            new_population = []
            for idx, current in enumerate(population):
                a, b, c = np.random.choice(population, 3, replace=False)
                mutant = mutate(current, a, b)
                crossover_points = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover_points, mutant, current)
                if func(trial) < fitness_values[idx]:
                    new_population.append(trial)
                else:
                    new_population.append(current)
            return np.array(new_population), pbest_idx, gbest_idx

        def PSO_Operator(population, fitness_values, pbest_idx, gbest_idx):
            velocities = np.zeros((self.population_size, self.dim))
            pbest = population[pbest_idx]
            gbest = population[gbest_idx]
            for _ in range(self.max_iter):
                for idx, current in enumerate(population):
                    r1, r2 = np.random.uniform(0, 1, (2, self.dim))
                    velocities[idx] = self.w * velocities[idx] + self.c1 * r1 * (pbest - current) + self.c2 * r2 * (gbest - current)
                    population[idx] = clipToBounds(current + velocities[idx])
                fitness_values = evaluate_population(population)
                pbest_idx, gbest_idx = DE_Operator(population, fitness_values)[1:]
                pbest = population[pbest_idx]
                gbest = population[gbest_idx]
            return gbest

        population = initialize_population()
        fitness_values = evaluate_population(population)
        pbest_idx, gbest_idx = DE_Operator(population, fitness_values)[1:]
        gbest = PSO_Operator(population, fitness_values, pbest_idx, gbest_idx)
        return gbest