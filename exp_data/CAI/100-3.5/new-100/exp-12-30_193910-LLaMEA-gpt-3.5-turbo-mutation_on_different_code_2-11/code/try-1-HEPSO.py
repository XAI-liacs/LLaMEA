class HEPSO:
    def __init__(self, budget=10000, dim=10, swarm_size=30, omega=0.5, phi_p=0.5, phi_g=0.5, omega_decay=0.99):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.omega = omega
        self.phi_p = phi_p
        self.phi_g = phi_g
        self.omega_decay = omega_decay
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        swarm = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.swarm_size, self.dim))
        velocities = np.zeros((self.swarm_size, self.dim))
        p_best = swarm.copy()
        f_vals = np.array([func(p) for p in swarm])
        p_best_vals = f_vals.copy()
        g_best_idx = np.argmin(f_vals)
        g_best = swarm[g_best_idx].copy()
        
        for _ in range(self.budget):
            for i in range(self.swarm_size):
                r_p = np.random.uniform(0, 1, size=self.dim)
                r_g = np.random.uniform(0, 1, size=self.dim)
                
                velocities[i] = self.omega * velocities[i] + self.phi_p * r_p * (p_best[i] - swarm[i]) + self.phi_g * r_g * (g_best - swarm[i])
                swarm[i] += velocities[i]
                
                f = func(swarm[i])
                if f < p_best_vals[i]:
                    p_best[i] = swarm[i].copy()
                    p_best_vals[i] = f
                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = swarm[i].copy()
                
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = swarm[i].copy()
                    g_best = swarm[i].copy()
            
            self.omega *= self.omega_decay
        
        return self.f_opt, self.x_opt