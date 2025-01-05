class EvolutionaryStrategy:
    def __call__(self, func):
        def mutate(x, sigma):
            return x + np.random.standard_cauchy(size=len(x)) * sigma