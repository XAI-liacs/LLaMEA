import numpy as np

class HybridGOA_DE_Enhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.c = 0.00001
        self.f_min, self.f_max = 0.00001, 1.0
        self.elite_rate = 0.2

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

            self.elite_rate = max(0.1, 0.3 * (1 - evaluations / self.budget))
            elite_count = max(1, int(self.population_size * self.elite_rate))
            elite_indices = np.argsort(fitness)[:elite_count]
            elite_solutions = population[elite_indices]

            for idx in elite_indices:
                if elite_count > 2:
                    a, b, c = elite_solutions[np.random.choice(elite_count, 3, replace=False)]
                    mutation_factor = 0.8 + 0.5 * np.random.rand() * np.std(population, axis=0).mean()  # Changed
                    mutant = np.clip(a + mutation_factor * (b - c) * 0.5, lb, ub)
                else:
                    a, b = elite_solutions[np.random.choice(elite_count, 2, replace=True)]
                    mutant = np.clip(a + 0.8 * (b - a), lb, ub)

                cr = 0.9 - (0.5 * (evaluations / self.budget))  # Changed
                trial = np.where(np.random.rand(self.dim) < cr, mutant, new_population[idx])
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < new_fitness[idx]:
                    new_population[idx] = trial
                    new_fitness[idx] = trial_fitness

            diversity = np.mean(np.std(population, axis=0))  # Calculate diversity
            for i in range(self.population_size):
                convergence_factor = np.mean(np.std(new_population, axis=0))  # Calculate convergence
                exploration_factor = 0.3 * (1 - evaluations / self.budget) * convergence_factor  # Changed
                if np.random.rand() < exploration_factor:
                    perturbation = np.random.uniform(-0.3, 0.3, self.dim) * diversity
                    candidate = np.clip(new_population[i] + perturbation, lb, ub)
                    candidate_fitness = func(candidate)
                    evaluations += 1
                    if candidate_fitness < new_fitness[i]:
                        new_population[i] = candidate
                        new_fitness[i] = candidate_fitness

            population = new_population
            fitness = new_fitness

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]