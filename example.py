import numpy as np
import numpy.random as rnd

from xp import dispatch
import xp.tools


def experiment(seed=None, method=None, N=None):
    # Integrate f(x) = x^2 over [0, 1]
    def f(x):
        return x**2

    rnd.seed(seed)

    if method == "stochastic":
        x = rnd.rand(N)
        estimate = np.mean(f(x))
    elif method == "deterministic":
        x = np.linspace(0, 1, N)
        y = f(x)
        estimate = np.trapz(y, x)
    else:
        raise ValueError("Unknown method")
    true_val = 1 / 3
    error = abs(estimate - true_val)
    return dict(estimate=estimate, true_val=true_val, error=error)


def list_experiments():
    xps = []
    # Use a loop with clauses for fine-grained control parameter config
    for method in ["stochastic", "deterministic"]:
        kws = {}  # overrule `common` params ⇒ dupes (to remove)
        if method == "deterministic":
            kws["seed"] = None
        xps.append(dict(method=method, **kws))

    # Convenience function to re-do each experiment for a list of common parameters.
    common = xp.tools.dict_prod(
        seed=3000 + np.arange(5),
        N=[10, 100, 1000],
    )
    # Combine: each xps item gets all common combinations
    xps = [{**c, **d} for c in common for d in xps]  # {common, xp} share any key ⇒ dupes
    xps = [dict(t) for t in {tuple(d.items()) for d in xps}]  # remove dupes
    return xps


if __name__ == "__main__":
    xps = list_experiments()
    # results = [experiment(**kwargs) for kwargs in xps]
    host = "localhost"  # or "my-gcp-*" or "cno-0001" or "hpc.intra.norceresearch" or None
    data_dir = dispatch(experiment, xps, host)
