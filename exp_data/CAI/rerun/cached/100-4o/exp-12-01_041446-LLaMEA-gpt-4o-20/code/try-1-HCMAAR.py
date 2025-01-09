import numpy as np

class HCMAAR:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.sigma = 0.3
        self.population_size = 4 + int(3 * np.log(dim))
        self.weights = np.log(self.population_size + 0.5) - np.log(np.arange(1, self.population_size + 1))
        self.weights /= np.sum(self.weights)
        self.mu_eff = 1. / np.sum(self.weights**2)
        self.covariance_matrix = np.eye(dim)
        self.mean = np.random.uniform(self.lower_bound, self.upper_bound, dim)
        self.lambda_ = self.population_size
        self.cs = (self.mu_eff + 2) / (dim + self.mu_eff + 5)
        self.ds = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (dim + 1)) - 1) + self.cs
        self.cc = (4 + self.mu_eff / dim) / (dim + 4 + 2 * self.mu_eff / dim)
        self.c1 = 2 / ((dim + 1.3)**2 + self.mu_eff)
        self.cmu = min(1 - self.c1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((dim + 2)**2 + self.mu_eff))
        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)
        self.de_factor = 0.5  # Differential Evolution factor
        self.cr = 0.9  # Crossover rate for DE

    def __call__(self, func):
        eval_count = 0
        best_solution = None
        best_fitness = float('inf')
        
        while eval_count < self.budget:
            population = np.random.multivariate_normal(self.mean, self.sigma**2 * self.covariance_matrix, self.lambda_)
            population = np.clip(population, self.lower_bound, self.upper_bound)
            fitness_values = np.array([func(ind) for ind in population])
            eval_count += self.lambda_

            indices = np.argsort(fitness_values)
            population = population[indices]
            fitness_values = fitness_values[indices]

            if fitness_values[0] < best_fitness:
                best_fitness = fitness_values[0]
                best_solution = population[0]

            for i in range(self.lambda_):
                a, b, c = np.random.choice(self.lambda_, 3, replace=False)
                mutant = np.clip(population[a] + self.de_factor * (population[b] - population[c]), self.lower_bound, self.upper_bound)
                trial = np.where(np.random.rand(self.dim) < self.cr, mutant, population[i])
                trial_fitness = func(trial)
                eval_count += 1
                if trial_fitness < fitness_values[i]:
                    population[i] = trial
                    fitness_values[i] = trial_fitness

            artmp = (population[:self.lambda_ // 2] - self.mean) / self.sigma
            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * np.mean(artmp, axis=0)
            hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2 * eval_count / self.lambda_)) < (1.4 + 2 / (self.dim + 1))
            self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * np.mean(artmp, axis=0)

            rank_one = np.outer(self.pc, self.pc)
            rank_mu = np.dot((self.weights * artmp.T), artmp)
            self.covariance_matrix = (1 - self.c1 - self.cmu) * self.covariance_matrix + self.c1 * rank_one + self.cmu * rank_mu

            self.sigma *= np.exp((self.cs / self.ds) * (np.linalg.norm(self.ps) / np.sqrt(self.dim) - 1))

            self.mean = np.dot(self.weights, population[:self.lambda_ // 2])
            
            if eval_count + self.population_size > self.budget:
                break

        return best_solution