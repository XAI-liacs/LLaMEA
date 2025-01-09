class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, adapt_scale=True):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.adapt_scale = adapt_scale
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.f_opt = np.Inf
        self.x_opt = None

    def mutate(self, target, population):
        candidates = np.copy(population)
        candidates = np.delete(candidates, target, axis=0)
        np.random.shuffle(candidates)
        a, b, c = candidates[:3]
        scale = np.random.uniform(0.1, 0.9) if self.adapt_scale else self.F
        mutant_vector = population[a] + scale * (population[b] - population[c])
        return mutant_vector