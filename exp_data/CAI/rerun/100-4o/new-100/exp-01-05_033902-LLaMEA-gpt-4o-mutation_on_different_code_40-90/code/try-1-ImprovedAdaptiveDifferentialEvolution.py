import numpy as np

class ImprovedAdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.bounds = (-5.0, 5.0)
        self.f_opt = np.Inf
        self.x_opt = None
        np.random.seed(42)  # For reproducibility

    def mutate(self, population, best_idx):
        indices = list(range(len(population)))
        indices.remove(best_idx)
        np.random.shuffle(indices)
        r1, r2, r3 = indices[:3]
        f = np.random.uniform(0.5, 1.0)  # Self-adaptive differential weight
        mutant_vector = (
            population[r1] 
            + f * (population[r2] - population[r3])
        )
        mutant_vector = np.clip(mutant_vector, *self.bounds)
        return mutant_vector

    def crossover(self, target_vector, mutant_vector):
        cr = np.random.uniform(0.5, 1.0)  # Self-adaptive crossover probability
        crossover_vector = np.array([
            mutant_vector[i] if np.random.rand() <= cr else target_vector[i]
            for i in range(self.dim)
        ])
        return crossover_vector

    def select(self, target_vector, trial_vector, func):
        target_fit = func(target_vector)
        trial_fit = func(trial_vector)
        return (trial_vector, trial_fit) if trial_fit < target_fit else (target_vector, target_fit)

    def reinitialize_population(self, population, fitness, best_idx):
        for i in range(self.population_size // 2):  # Reinitialize half of the population
            if i == best_idx:
                continue
            population[i] = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)
            fitness[i] = func(population[i])

    def __call__(self, func):
        population = np.random.uniform(
            self.bounds[0], self.bounds[1], size=(self.population_size, self.dim)
        )
        fitness = np.array([func(indiv) for indiv in population])
        best_idx = np.argmin(fitness)
        
        eval_count = self.population_size

        while eval_count < self.budget:
            if eval_count % (self.population_size * 5) == 0:  # Reinitialize every few generations
                self.reinitialize_population(population, fitness, best_idx)
            
            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break

                mutant_vector = self.mutate(population, best_idx)
                trial_vector = self.crossover(population[i], mutant_vector)
                population[i], fitness[i] = self.select(population[i], trial_vector, func)
                eval_count += 1

                if fitness[i] < self.f_opt:
                    self.f_opt = fitness[i]
                    self.x_opt = population[i]
                    best_idx = i

        return self.f_opt, self.x_opt