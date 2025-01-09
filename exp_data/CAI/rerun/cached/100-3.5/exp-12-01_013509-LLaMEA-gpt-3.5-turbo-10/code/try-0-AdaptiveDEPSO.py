import numpy as np

class AdaptiveDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.max_iter = budget // self.population_size
        
    def __call__(self, func):
        def mutation(population, f=0.5):
            idxs = np.arange(self.population_size)
            while True:
                r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
                mutant = population[r1] + f * (population[r2] - population[r3])
                yield mutant
        
        def crossover(target, mutant, cr=0.9):
            crossover_points = np.random.rand(self.dim) < cr
            trial = np.where(crossover_points, mutant, target)
            return trial
        
        def pso_update(particle, pbest, gbest, w=0.5, c1=1.5, c2=1.5):
            inertia = w * particle
            cognitive = c1 * np.random.rand(self.dim) * (pbest - particle)
            social = c2 * np.random.rand(self.dim) * (gbest - particle)
            return inertia + cognitive + social
        
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        pbest = population.copy()
        gbest = population[np.argmin(fitness)]
        
        for _ in range(self.max_iter):
            for idx, particle in enumerate(population):
                mutant_gen = mutation(population)
                mutant = next(mutant_gen)
                trial = crossover(particle, mutant)
                trial_fitness = func(trial)
                if trial_fitness < fitness[idx]:
                    population[idx] = trial
                    fitness[idx] = trial_fitness
                    pbest[idx] = trial
                    if trial_fitness < func(gbest):
                        gbest = trial
            
            for idx, particle in enumerate(population):
                updated_particle = pso_update(particle, pbest[idx], gbest)
                population[idx] = np.clip(updated_particle, -5.0, 5.0)
        
        return gbest