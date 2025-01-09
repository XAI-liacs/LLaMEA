import numpy as np

class QuantumEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        
        for _ in range(self.budget):
            quantum_population = np.abs(np.fft.fft(population, axis=0))
            fitness = np.array([func(ind) for ind in quantum_population])
            selected_indices = np.argsort(fitness)[:self.budget//2]
            selected_population = quantum_population[selected_indices]
            mutated_population = selected_population + np.random.normal(0, 0.1, size=selected_population.shape)
            population[selected_indices] = np.random.choice(mutated_population, size=self.budget//2, replace=False)

        best_solution = population[np.argmin([func(ind) for ind in population])]
        return best_solution