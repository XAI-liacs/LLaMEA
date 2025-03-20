import numpy as np

class HybridGOA_DE_Enhanced:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.c = 0.00001
        self.f_min, self.f_max = 0.00001, 1.0

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

            elite_rate = max(0.1, 0.3 * (1 - evaluations / self.budget))
            elite_count = max(1, int(self.population_size * elite_rate))
            elite_indices = np.argsort(fitness)[:elite_count]
            elite_solutions = population[elite_indices]

            for idx in elite_indices:
                if elite_count > 2:
                    a, b, c = elite_solutions[np.random.choice(elite_count, 3, replace=False)]
                    mutant = np.clip(a + (0.8 + 0.4 * np.random.rand()) * (b - c), lb, ub)
                else:
                    a, b = elite_solutions[np.random.choice(elite_count, 2, replace=True)]
                    mutant = np.clip(a + 0.8 * (b - a), lb, ub)

                cr = 0.9 - (0.7 * (evaluations / self.budget))
                trial = np.where(np.random.rand(self.dim) < cr, mutant, new_population[idx])
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < new_fitness[idx]:
                    new_population[idx] = trial
                    new_fitness[idx] = trial_fitness

            if np.random.rand() < 0.5 * (1 - evaluations / self.budget):
                for i in range(self.population_size):
                    perturbation = np.random.uniform(-0.15, 0.15, self.dim)
                    candidate = np.clip(new_population[i] + perturbation, lb, ub)
                    candidate_fitness = func(candidate)
                    evaluations += 1
                    if candidate_fitness < new_fitness[i]:
                        new_population[i] = candidate
                        new_fitness[i] = candidate_fitness

            if evaluations < self.budget * 0.75:  # Dynamic population management
                self.population_size = min(50, self.population_size + 1)
                additional_population = np.random.uniform(lb, ub, (1, self.dim))
                new_population = np.vstack((new_population, additional_population))
                new_fitness = np.append(new_fitness, func(additional_population[0]))
                evaluations += 1

            population = new_population
            fitness = new_fitness

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]