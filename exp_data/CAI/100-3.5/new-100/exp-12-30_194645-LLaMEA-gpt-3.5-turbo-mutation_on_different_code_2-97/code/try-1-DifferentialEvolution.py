class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=None, CR=0.9, pop_size=50):
        self.budget = budget
        self.dim = dim
        self.F = np.random.uniform(0.1, 0.9) if F is None else F
        self.CR = CR
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None