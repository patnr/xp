import pathos.multiprocessing as MP
import threadpoolctl
from tqdm.auto import tqdm

bar_frmt = "{l_bar}|{bar}| {n_fmt}/{total_fmt}, ⏱️ {elapsed} ⏳{remaining}, {rate_fmt}{postfix}"
threadpoolctl.threadpool_limits(1)  # make np use only 1 core


def progbar(*args, **kwargs):
    return tqdm(*args, bar_format=bar_frmt, **kwargs)


def mp(f, lst, nCPU=None):
    """Multiprocessing map with progress bar."""
    if nCPU in [None, "all"] or nCPU is True:
        nCPU = MP.cpu_count()

    if nCPU in [0, 1, False]:
        # Use this for debugging
        jobs = map(f, lst)
    else:
        # Chunking is important for speed, but not done automatically by imap.
        D = 1 + len(lst) // nCPU // 10  # heuristic chunksize
        with MP.ProcessPool(nCPU) as pool:
            jobs = pool.imap(f, lst, chunksize=D)
    return list(progbar(jobs, total=len(lst)))
