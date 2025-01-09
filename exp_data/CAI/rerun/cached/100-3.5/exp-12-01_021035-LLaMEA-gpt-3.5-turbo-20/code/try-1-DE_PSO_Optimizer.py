import numpy as np

class DE_PSO_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.max_iters = budget // self.pop_size
        self.cr = 0.9
        self.f = 0.8
        self.w = 0.5
        self.c1 = 1.496
        self.c2 = 1.496

    def __call__(self, func):
        def random_init_population():
            return np.random.uniform(low=-5.0, high=5.0, size=(self.pop_size, self.dim))

        def evaluate_population(population):
            return np.array([func(individual) for individual in population])

        def mutate(parents, target_idx):
            r1, r2, r3 = np.random.choice(len(parents), 3, replace=False)
            mutant = parents[r1] + self.f * (parents[r2] - parents[r3])
            crossover = np.random.rand(self.dim) < self.cr
            trial = np.where(crossover, mutant, parents[target_idx])
            return trial

        def update_velocity_velocity(position, velocity, pbest, gbest):
            cognitive = self.c1 * np.random.rand(self.dim) * (pbest - position)
            social = self.c2 * np.random.rand(self.dim) * (gbest - position)
            return self.w * velocity + cognitive + social

        population = random_init_population()
        fitness = evaluate_population(population)
        pbest = population.copy()
        gbest = population[np.argmin(fitness)]
        velocity = np.zeros((self.pop_size, self.dim))

        for _ in range(self.max_iters):
            for i in range(self.pop_size):
                trial = mutate(population, i)
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    pbest[i] = trial
                if trial_fitness < func(gbest):
                    gbest = trial

                velocity[i] = update_velocity_velocity(population[i], velocity[i], pbest[i], gbest)
                population[i] += velocity[i]

        return gbest