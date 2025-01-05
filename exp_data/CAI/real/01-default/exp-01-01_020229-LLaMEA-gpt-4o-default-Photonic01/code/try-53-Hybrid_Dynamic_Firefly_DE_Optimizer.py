import numpy as np

class Hybrid_Dynamic_Firefly_DE_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.alpha_0 = 0.5
        self.beta_0 = 1.0
        self.gamma = 1.0
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7
        self.adaptive_decay = 0.99

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)

        # Initialize population
        position = np.random.uniform(lb, ub, (self.population_size, self.dim))
        light_intensity = np.array([func(p) for p in position])
        
        evaluations = self.population_size
        global_best_position = position[np.argmin(light_intensity)]
        global_best_value = np.min(light_intensity)

        while evaluations < self.budget:
            # Dynamic updating of alpha and mutation factor
            alpha = self.alpha_0 * (self.adaptive_decay ** (evaluations / self.population_size))
            mutation_factor = self.mutation_factor * (1 + np.sin(evaluations / self.budget * np.pi))

            for i in range(self.population_size):
                for j in range(self.population_size):
                    if light_intensity[i] > light_intensity[j]:
                        distance = np.linalg.norm(position[i] - position[j])
                        beta = self.beta_0 * np.exp(-self.gamma * distance ** 2)
                        attraction = beta * (position[j] - position[i]) + alpha * np.random.uniform(-1, 1, self.dim)
                        position[i] += attraction
                        position[i] = np.clip(position[i], lb, ub)

                # Apply Differential Evolution (DE) strategy
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = a + mutation_factor * (b - c)
                trial = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant, position[i])
                trial = np.clip(trial, lb, ub)

                trial_value = func(trial)
                evaluations += 1

                if trial_value < light_intensity[i]:
                    position[i] = trial
                    light_intensity[i] = trial_value

                if trial_value < global_best_value:
                    global_best_position = trial
                    global_best_value = trial_value

                if evaluations >= self.budget:
                    break

        return global_best_position, global_best_value