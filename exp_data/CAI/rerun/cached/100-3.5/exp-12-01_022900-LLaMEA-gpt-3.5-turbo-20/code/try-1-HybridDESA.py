class HybridDESA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Unchanged code for HybridDESA optimization