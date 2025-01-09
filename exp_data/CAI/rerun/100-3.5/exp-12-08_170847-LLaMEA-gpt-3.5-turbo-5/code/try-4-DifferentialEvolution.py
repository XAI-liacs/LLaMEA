class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, pop_size=50):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.pop_size = max(50, int(budget / 200))  # Dynamic population size
        self.f_opt = np.Inf
        self.x_opt = None