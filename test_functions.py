import numpy as np
from dataclasses import dataclass
import json

@dataclass
class SeparabilityReport:
    separable: bool
    percent_noncompliance: float
    mean_norm_interaction: float
    max_norm_interaction: float
    per_pair_score: dict  # {(i,j): score}
    details: dict         # misc diagnostics

def evaluate_separability(f, dim, bounds=(-5.0, 5.0), samples=128, h=None,
                          tol=1e-6, rng=None):
    """
    Black-box separability test for f: R^dim -> R.
    Uses finite-difference Hessian cross-terms + a superposition check.

    Parameters
    ----------
    f : callable
        Function taking 1D np.ndarray (len=dim) -> float.
    dim : int
        Dimensionality.
    bounds : tuple(float, float)
        Box to sample from (same for all dims).
    samples : int
        Number of random points for finite-difference probing.
    h : float or None
        Step size for finite differences. If None, set to 1e-3 * (ub-lb).
    tol : float
        Threshold on normalized interaction to count as non-compliant.
    rng : np.random.Generator or None
        For reproducibility.

    Returns
    -------
    SeparabilityReport
    """
    if rng is None:
        rng = np.random.default_rng()
    lb, ub = map(float, bounds)
    box = ub - lb
    if h is None:
        h = 1e-3 * box
    eps = 1e-12

    # Draw sample points inside the box
    X = rng.uniform(lb + 2*h, ub - 2*h, size=(samples, dim))

    # Helpers for finite differences
    def e(i):
        v = np.zeros(dim); v[i] = 1.0
        return v

    def f_at(x):
        return float(f(x))

    def d2_ij(x, i, j):
        """Central difference mixed second derivative ∂^2 f / ∂x_i ∂x_j."""
        ei, ej = e(i), e(j)
        return (
            f_at(x + h*ei + h*ej) - f_at(x + h*ei - h*ej)
            - f_at(x - h*ei + h*ej) + f_at(x - h*ei - h*ej)
        ) / (4*h*h)

    def d2_ii(x, i):
        """Central difference second derivative ∂^2 f / ∂x_i^2."""
        ei = e(i)
        return (f_at(x + h*ei) - 2.0*f_at(x) + f_at(x - h*ei)) / (h*h)

    # Estimate diagonal curvature scales per dim for normalization
    Hii_vals = np.zeros((samples, dim))
    for s in range(samples):
        x = X[s]
        fx = f_at(x)  # cache center eval to reduce calls on diagonal calc
        for i in range(dim):
            ei = np.zeros(dim); ei[i] = 1.0
            Hii_vals[s, i] = (f_at(x + h*ei) - 2.0*fx + f_at(x - h*ei)) / (h*h)

    # Robust scale per axis: median absolute curvature (avoid zero division)
    scale_i = np.median(np.abs(Hii_vals), axis=0) + eps

    # Compute normalized cross interactions
    pair_scores = {}  # aggregated per pair over samples (median of |H_ij|/sqrt(scale_i*scale_j))
    all_norm_ijs = []  # for overall stats
    per_pair_all = { (i,j): [] for i in range(dim) for j in range(i+1, dim) }

    for s in range(samples):
        x = X[s]
        # Precompute f(x) once for superposition later
        for i in range(dim):
            pass  # nothing extra here; left for symmetry

        for i in range(dim):
            for j in range(i+1, dim):
                Hij = d2_ij(x, i, j)
                norm = abs(Hij) / np.sqrt(scale_i[i]*scale_i[j])
                per_pair_all[(i,j)].append(norm)
                all_norm_ijs.append(norm)

    for (i,j), arr in per_pair_all.items():
        pair_scores[(i,j)] = float(np.median(arr))

    mean_norm_interaction = float(np.mean(all_norm_ijs)) if all_norm_ijs else 0.0
    max_norm_interaction = float(np.max(all_norm_ijs)) if all_norm_ijs else 0.0

    # Superposition sanity check: does Δ_i depend on j?
    # If separable, Δ_i(x) ≈ f(x+he_i)-f(x) should not change when j is perturbed.
    sup_violations = 0
    sup_total = 0
    for s in range(samples):
        x = X[s]
        fx = f_at(x)
        for i in range(dim):
            ei = e(i)
            d_i = f_at(x + h*ei) - fx
            for j in range(dim):
                if j == i: continue
                ej = e(j)
                d_i_with_j = f_at(x + h*ej + h*ei) - f_at(x + h*ej)
                # Relative interaction score (bounded, scale-free-ish)
                denom = abs(d_i) + abs(d_i_with_j) + eps
                rel = abs(d_i_with_j - d_i) / denom
                sup_total += 1
                if rel > tol:
                    sup_violations += 1

    # Combine both notions into a single % non-compliance.
    # 50/50 weight between Hessian cross-terms and superposition violations.
    # For Hessian: count fraction above tol
    hess_violations = np.sum(np.array(all_norm_ijs) > tol)
    hess_total = len(all_norm_ijs)
    frac_hess = (hess_violations / max(1, hess_total))
    frac_sup = (sup_violations / max(1, sup_total))
    percent_noncompliance = 100.0 * (0.5*frac_hess + 0.5*frac_sup)

    separable = percent_noncompliance < 0.5  # basically “all but numerical noise”

    details = {
        "tol": tol,
        "h": h,
        "samples": samples,
        "bounds": (lb, ub),
        "diag_curvature_scale_per_dim": scale_i.tolist(),
        "hessian_fraction_violations": frac_hess,
        "superposition_fraction_violations": frac_sup,
    }

    return SeparabilityReport(
        separable=separable,
        percent_noncompliance=float(percent_noncompliance),
        mean_norm_interaction=mean_norm_interaction,
        max_norm_interaction=max_norm_interaction,
        per_pair_score={str(k): v for k,v in pair_scores.items()},
        details=details,
    )

# load jsonl file and evaluate separability for each function
# load jsonl file
def load_jsonl(file_path):
    """
    Load a JSONL file and return a list of dictionaries.
    Each line in the file should be a valid JSON object.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # skip empty lines
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # skip invalid JSON lines
    return data

log = load_jsonl("/home/neocortex/repos/LLaMEA-ELA/LLaMEA/exp-08-08_093639-LLaMEA-gpt-5-nano-ELA-basins_scaled_separable_scaled/log.jsonl")
for entry in log:
    if 'code' in entry:
        code = entry['code']
        algorithm_name = entry['name']
        exec(code, globals())

        algorithm = None
        # Final validation
        dim = 5 #change to appropriate dimensionality
        algorithm = globals()[algorithm_name](dim)
        f = algorithm.f

        bounds = (-5.0, 5.0)
        report = evaluate_separability(f, dim, bounds=bounds, samples=1024)
        print(f"Function: {algorithm_name}, Separable: {report.separable}")