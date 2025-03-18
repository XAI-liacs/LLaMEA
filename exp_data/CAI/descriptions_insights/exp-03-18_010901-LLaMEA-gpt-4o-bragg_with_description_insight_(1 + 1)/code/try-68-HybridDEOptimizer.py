class HybridDEOptimizer:
    # ... (rest of the code remains unchanged)

    def differential_evolution(self, func, bounds):
        # ... (previous code remains unchanged)
        
        while self.current_evals < self.budget:
            self.population_size = max(5, self.population_size - (self.current_evals // (self.budget // 5)))
            for i in range(self.population_size):
                if self.current_evals >= self.budget:
                    break
                
                # ... (previous code remains unchanged)

                adaptive_period = 1 + (self.current_evals // (self.budget // 4))  # More adaptive periodicity
                trial = self.ensure_periodicity(trial, period=adaptive_period)

                # ... (rest of the code remains unchanged)
                
        return best_solution

    def local_refinement(self, func, best_solution, bounds):
        if np.random.rand() < 0.6:  # Increased chance for local refinement
            result = minimize(func, best_solution, bounds=[(bounds.lb[i], bounds.ub[i]) for i in range(self.dim)], method='L-BFGS-B')
            return result.x
        return best_solution