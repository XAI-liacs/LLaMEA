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
                stochastic_perturbation = np.random.randn(self.dim) * 0.05  # Added stochastic perturbation
                new_position = c_i * s_i + population[i] + stochastic_perturbation
                new_population[i] = np.clip(new_position, lb, ub)

            new_fitness = np.array([func(ind) for ind in new_population])
            evaluations += self.population_size

            self.elite_rate = max(0.15, 0.2 * (1 - evaluations / self.budget))  # Increased minimum elite rate
            elite_count = max(2, int(self.population_size * self.elite_rate))  # Ensured more elite members
            elite_indices = np.argsort(fitness)[:elite_count]
            elite_solutions = population[elite_indices]

            for idx in elite_indices:
                a, b, c = elite_solutions[np.random.choice(elite_count, 3, replace=False)]
                mutant = np.clip(a + 0.9 * (b - c), lb, ub)  # Increased mutation factor

                cr = 0.75 - (0.6 * (evaluations / self.budget))  # Adjusted crossover rate
                trial = np.where(np.random.rand(self.dim) < cr, mutant, new_population[idx])
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < new_fitness[idx]:
                    new_population[idx] = trial
                    new_fitness[idx] = trial_fitness
            
            for i in range(self.population_size):
                if np.random.rand() < 0.35:  # Increased perturbation chance
                    perturbation = np.random.uniform(-0.3, 0.3, self.dim)  # Adjusted perturbation range
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