import numpy as np

class AdaptiveCuckooSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.step_size = 0.1  # Initial step size

    def levy_flight(self):
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma)
        v = np.random.normal(0, 1)
        step = u / abs(v) ** (1 / beta)
        return step

    def update_step_size(self, diversity):
        self.step_size *= np.exp(-diversity)

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitness = np.array([func(x) for x in population])
        
        for _ in range(self.budget):
            diversity = np.mean(np.std(population, axis=0))
            self.update_step_size(diversity)
            new_population = []
            
            for i in range(self.population_size):
                cuckoo = population[i]
                step = self.step_size * self.levy_flight()
                random_cuckoo = population[np.random.randint(0, self.population_size)]
                new_cuckoo = cuckoo + step * (cuckoo - random_cuckoo)
                
                if func(new_cuckoo) < fitness[i]:
                    population[i] = new_cuckoo
                    fitness[i] = func(new_cuckoo)
                    
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]
        
        return best_solution