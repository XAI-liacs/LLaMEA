import numpy as np

class HybridSineCosineGravitationalSearchOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.current_evaluations = 0
        self.G0 = 100  # Initial gravitational constant
        self.alpha = 0.7  # Control parameter for sine-cosine mechanism

    def __call__(self, func):
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        population = np.random.rand(self.population_size, self.dim) * (bounds[:,1] - bounds[:,0]) + bounds[:,0]
        fitness = np.array([func(ind) for ind in population])
        best_index = np.argmin(fitness)
        best_position = np.copy(population[best_index])
        self.current_evaluations += self.population_size
        
        while self.current_evaluations < self.budget:
            G = self.G0 * (1 - self.current_evaluations / self.budget)  # Decaying gravitational constant
            mass = 1 / (fitness - fitness.min() + 1e-10)
            mass /= mass.sum()
            
            # Update velocities and positions
            for i in range(self.population_size):
                force = np.zeros(self.dim)
                for j in range(self.population_size):
                    if i != j:
                        dist = np.linalg.norm(population[j] - population[i])
                        force += G * mass[j] * (population[j] - population[i]) / (dist + 1e-10)
                acceleration = force * mass[i]
                
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                population[i] += self.alpha * np.sin(r1) * np.abs(acceleration) + self.alpha * np.cos(r2) * np.abs(best_position - population[i])
                population[i] = np.clip(population[i], bounds[:,0], bounds[:,1])
                
                current_fitness = func(population[i])
                self.current_evaluations += 1
                
                if current_fitness < fitness[i]:
                    fitness[i] = current_fitness
                    if current_fitness < fitness[best_index]:
                        best_position = population[i]
                        best_index = i
            
            if self.current_evaluations >= self.budget:
                break
            
        return best_position, fitness[best_index]