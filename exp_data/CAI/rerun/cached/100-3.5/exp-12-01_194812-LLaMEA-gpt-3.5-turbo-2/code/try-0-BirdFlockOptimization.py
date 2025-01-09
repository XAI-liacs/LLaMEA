import numpy as np

class BirdFlockOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def move_towards_best(bird_position, best_position, step_size):
            return bird_position + step_size * (best_position - bird_position)

        population = np.random.uniform(-5.0, 5.0, size=(self.budget, self.dim))
        best_position = population[np.argmin([func(ind) for ind in population])]
        
        for _ in range(self.budget):
            step_size = np.random.uniform(0, 1)
            population = np.array([move_towards_best(bird, best_position, step_size) for bird in population])
            best_position = population[np.argmin([func(ind) for ind in population])]

        return best_position