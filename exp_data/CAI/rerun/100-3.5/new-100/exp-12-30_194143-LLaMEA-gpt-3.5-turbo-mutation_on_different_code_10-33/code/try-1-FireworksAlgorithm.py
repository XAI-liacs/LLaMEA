class FireworksAlgorithm:
    def __init__(self, budget=10000, dim=10, num_sparks=5):
        self.budget = budget
        self.dim = dim
        self.num_sparks = max(5, int(0.1 * self.budget))
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        for i in range(self.budget):
            sparks = [np.random.uniform(func.bounds.lb, func.bounds.ub, size=self.dim) for _ in range(self.num_sparks)]
            
            for spark in sparks:
                f = func(spark)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = spark
                    
        return self.f_opt, self.x_opt