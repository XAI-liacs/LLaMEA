import numpy as np

class HybridFireflyDE:
    def __init__(self, budget=10000, dim=10, population_size=20, alpha=0.5, beta_min=0.2, gamma=1.0, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.alpha = alpha
        self.beta_min = beta_min
        self.gamma = gamma
        self.F = F
        self.CR = CR

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        
        # Initialize population
        population = np.random.uniform(-100, 100, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        evals = self.population_size

        while evals < self.budget:
            # Firefly movement
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if fitness[j] < fitness[i]:
                        r = np.linalg.norm(population[i] - population[j])
                        beta = self.beta_min * np.exp(-self.gamma * r ** 2)
                        attraction = beta * (population[j] - population[i])
                        # Adjust alpha based on current best fitness
                        randomization = (self.alpha * (1 - evals / self.budget)) * (np.random.uniform(-1, 1, self.dim))
                        new_position = population[i] + attraction + randomization
                        new_position = np.clip(new_position, -100, 100)
                        new_fitness = func(new_position)
                        evals += 1
                        if new_fitness < fitness[i]:
                            population[i] = new_position
                            fitness[i] = new_fitness
                            if new_fitness < self.f_opt:
                                self.f_opt = new_fitness
                                self.x_opt = new_position
                        if evals >= self.budget:
                            break
                if evals >= self.budget:
                    break

            if evals >= self.budget:
                break
            
            # Differential Evolution
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = np.clip(x1 + self.F * (x2 - x3), -100, 100)
                trial = np.copy(population[i])
                crossover = np.random.rand(self.dim) < self.CR
                trial[crossover] = mutant[crossover]
                trial_fitness = func(trial)
                evals += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial
                if evals >= self.budget:
                    break
        
        return self.f_opt, self.x_opt