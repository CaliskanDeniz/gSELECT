import os
import platform
import psutil
import logging
from itertools import combinations


def get_memory_usage_mb():
    """
    Get the current memory usage of the process in megabytes (MB).

    Produces:
    • Current memory usage in MB.

    Parameters
    ----------
    None

    Returns
    -------
    float
        Current memory usage in MB.
    """
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss  # in bytes
    return mem / (1024 ** 2)  # Convert to MB


def log_peak_memory_usage():
    """
    Log the peak memory usage of the current process.

    Produces:
    • Logs peak RAM usage in megabytes (MB), adapted to OS.

    Parameters
    ----------
    None
    """
    system = platform.system()
    
    try:
        if system == "Windows":
            import psutil
            process = psutil.Process(os.getpid())
            peak_bytes = process.memory_info().peak_wset
            peak_mb = peak_bytes / (1024 ** 2)
        else:
            import resource
            peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak_mb = peak_kb / 1024  # Convert to MB
        logging.info(f"Peak memory usage: {peak_mb:.2f} MB")
    except Exception as e:
        logging.warning(f"Could not determine peak memory usage: {e}")


def get_unique_filename(filepath):
    """
    Generate a unique filename if the given file already exists.

    Produces:
    • Modified file path with a numerical suffix to avoid overwriting.

    Parameters
    ----------
    filepath : str
        Original file path.

    Returns
    -------
    str
        Unique file path.
    """
    if not os.path.exists(filepath):
        return filepath 

    base, ext = os.path.splitext(filepath)
    counter = 1
    new_filepath = f"{base}_{counter}{ext}"

    while os.path.exists(new_filepath):
        counter += 1
        new_filepath = f"{base}_{counter}{ext}"

    return new_filepath


def generate_explorative_gene_selections(n):
    """
    Generate all possible non-empty subsets of gene indices.

    Produces:
    • List of all possible gene subsets for exploration.

    Parameters
    ----------
    n : int
        Total number of genes to consider.

    Returns
    -------
    list of list
        All possible non-empty subsets of gene indices.
    """
    all_combinations = []
    elements = list(range(1, n+1))
    for r in range(1, n+1):
        all_combinations.extend(combinations(elements, r))
    return [list(comb) for comb in all_combinations]