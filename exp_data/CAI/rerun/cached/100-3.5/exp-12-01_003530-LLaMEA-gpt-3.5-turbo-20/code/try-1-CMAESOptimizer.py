import cma

class CMAESOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        es = cma.CMAEvolutionStrategy(np.random.rand(self.dim), 0.5)
        while not es.stop():
            solutions = es.ask()
            values = [func(sol) for sol in solutions]
            es.tell(solutions, values)
        return es.result[0]