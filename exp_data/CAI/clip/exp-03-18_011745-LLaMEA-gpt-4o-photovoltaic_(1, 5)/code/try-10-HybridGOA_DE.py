import numpy as np

class HybridGOA_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20  # Number of grasshoppers
        self.c = 0.00001  # Convergence constant
        self.f_min, self.f_max = 0.00001, 1.0  # Attraction intensity range
        self.elite_rate = 0.2  # The proportion of elite solutions for DE

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            f = self.f_max - evaluations / self.budget * (self.f_max - self.f_min)
            c_i = self.c * np.power(f, 2)

            new_population = np.zeros_like(population)
            for i in range(self.population_size):
                s_i = np.zeros(self.dim)
                for j in range(self.population_size):
                    if i != j:
                        r_ij = np.linalg.norm(population[j] - population[i])
                        s_ij = (population[j] - population[i]) * np.exp(-r_ij/self.f_max)
                        s_i += s_ij
                new_position = c_i * s_i + population[i]
                new_population[i] = np.clip(new_position, lb, ub)

            new_fitness = np.array([func(ind) for ind in new_population])
            evaluations += self.population_size

            # Differential Evolution step on the top elite_rate percent solutions
            elite_count = max(1, int(self.population_size * self.elite_rate))
            elite_indices = np.argsort(fitness)[:elite_count]
            elite_solutions = population[elite_indices]
            for idx in elite_indices:
                a, b, c = elite_solutions[np.random.choice(elite_count, 3, replace=False)]
                mutant = np.clip(a + 0.85 * (b - c), lb, ub)  # Modified scaling factor from 0.9 to 0.85
                trial = np.where(np.random.rand(self.dim) < 0.9, mutant, new_population[idx])
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < new_fitness[idx]:
                    new_population[idx] = trial
                    new_fitness[idx] = trial_fitness

            # Update population and fitness
            population = new_population
            fitness = new_fitness

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]