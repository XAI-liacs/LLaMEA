import numpy as np
from pyevolve import BaseOptimizer, Real, Minimize, Stats
from pyevolve import mutation, selection, operator
import random

class ProbabilisticMutationOperator(BaseOptimizer):
    def __init__(self, budget, dim):
        super().__init__()
        self.budget = budget
        self.dim = dim
        self.stats = Stats()
        self.real = Real(['x'], [-5.0, 5.0], self.dim)

    def __call__(self, func):
        self.pop = self.real.create()
        self.stats.clear()
        self.mutator = mutation.MutationOperator(mutation.GaussianMutation, mu=0.1, sigma=0.1)
        self.selector = selection.Selector(selection.TournamentSelector, tourn_size=5)
        self.operator = operator.Operators(['mutate','select'])
        self.run(self.budget, func)

        # Probabilistic line search
        for i in range(self.budget):
            if random.random() < 0.1:
                # Select a random individual from the population
                individual = random.choice(self.pop)
                # Calculate the gradient of the function at the individual's position
                gradient = np.zeros(self.dim)
                for j in range(self.dim):
                    gradient[j] = (func(individual[j] + 1e-6) - func(individual[j] - 1e-6)) / (2 * 1e-6)
                # Update the individual's position based on the gradient
                individual = individual + 0.1 * gradient

# Usage
if __name__ == "__main__":
    optimizer = ProbabilisticMutationOperator(budget=50, dim=10)
    for i in range(24):
        func = eval(f'lambda x: x**2 + {i}')
        optimizer(func)
        print(f"Function {i+1} optimized: {optimizer.best().fitness.value}")