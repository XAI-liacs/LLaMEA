import numpy as np

class Levy_Firefly_CMA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.alpha = 0.2
        self.gamma = 1.0
        self.beta_min = 0.2
        self.beta_max = 1.0
        self.sigma = 0.5

    def levy_flight(self, scale, size):
        beta = 1.5
        sigma = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                (np.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, size=size)
        v = np.random.normal(0, 1, size=size)
        step = u / np.abs(v) ** (1 / beta)
        return scale * step

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        np.random.seed(42)
        
        position = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(p) for p in position])
        
        global_best_position = position[np.argmin(fitness)]
        global_best_value = np.min(fitness)
        
        evaluations = self.population_size
        mean = np.mean(position, axis=0)
        cov = np.cov(position, rowvar=False) + np.eye(self.dim) * 1e-5

        while evaluations < self.budget:
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if fitness[i] > fitness[j]:
                        r = np.linalg.norm(position[i] - position[j])
                        beta = self.beta_min + (self.beta_max - self.beta_min) * np.exp(-self.gamma * r ** 2)
                        e = self.alpha * (np.random.rand(self.dim) - 0.5)
                        position[i] += beta * (position[j] - position[i]) + e
                        position[i] = np.clip(position[i], lb, ub)
                        position[i] += self.levy_flight(0.01, self.dim)
                        position[i] = np.clip(position[i], lb, ub)

                current_value = func(position[i])
                evaluations += 1
                
                if current_value < fitness[i]:
                    fitness[i] = current_value

                if current_value < global_best_value:
                    global_best_position = position[i]
                    global_best_value = current_value

                if evaluations >= self.budget:
                    break
            
            offspring = np.random.multivariate_normal(mean, cov, self.population_size)
            offspring = np.clip(offspring, lb, ub)
            offspring_fitness = np.array([func(o) for o in offspring])
            evaluations += self.population_size

            combined_position = np.vstack((position, offspring))
            combined_fitness = np.hstack((fitness, offspring_fitness))
            best_indices = np.argsort(combined_fitness)[:self.population_size]
            position = combined_position[best_indices]
            fitness = combined_fitness[best_indices]
            
            mean = np.mean(position, axis=0)
            cov = np.cov(position, rowvar=False) + np.eye(self.dim) * 1e-5
            
            if evaluations >= self.budget:
                break

        return global_best_position, global_best_value