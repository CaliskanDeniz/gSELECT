import os
import platform
import psutil
import logging
from itertools import combinations


def get_memory_usage_mb():
    """
    Get the current memory usage of the process in megabytes (MB).

    This function uses the `psutil` library to inspect the resident set size (RSS) 
    of the current Python process. RSS represents the portion of memory occupied 
    in RAM (excluding swapped out pages).

    Returns:
    --------
    float
        The current memory usage in megabytes (MB).
    """
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss  # in bytes
    return mem / (1024 ** 2)  # Convert to MB


def log_peak_memory_usage():
    """
    Log the peak memory usage of the current process, adapted to OS.

    This function retrieves the peak resident set size (RSS) of the process, 
    representing the highest RAM usage during its lifetime.

    Notes:
    ------
    - On Linux/macOS, uses `resource.getrusage()`, reported in kilobytes.
    - On Windows, uses `psutil` to get `.peak_wset`, reported in bytes.

    Logs:
    -----
    Peak memory usage in megabytes (MB).
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
    Generate a unique filename if a file with the given name already exists.
    
    This function appends a numerical counter to the filename to prevent overwriting
    existing files. It ensures that each new file gets a unique name.
    
    Example:
    --------
    If "output.csv" exists, the function will return "output_1.csv",
    then "output_2.csv", etc.
    
    Parameters:
    -----------
    filepath : str
        The original file path.
    
    Returns:
    --------
    str
        A modified file path that is unique.
    """
    if not os.path.exists(filepath):
        return filepath  # If file does not exist, use the original name

    base, ext = os.path.splitext(filepath)
    counter = 1
    new_filepath = f"{base}_{counter}{ext}"

    while os.path.exists(new_filepath):
        counter += 1
        new_filepath = f"{base}_{counter}{ext}"

    return new_filepath


def generate_explorative_gene_selections(n):
    """
    Generate all possible subsets of gene selections.
    
    This function creates all possible subsets of genes using itertools.combinations.
    It helps in evaluating different combinations of genes for selection.
    
    Example:
    --------
    If `n=3`, the function returns:
    [[1], [2], [3], [1,2], [1,3], [2,3], [1,2,3]]
    
    Parameters:
    -----------
    n : int
        The total number of genes to consider.
    
    Returns:
    --------
    list of list
        A list containing all possible subsets of gene indices.
    """
    all_combinations = []
    elements = list(range(1, n+1))
    for r in range(1, n+1):
        all_combinations.extend(combinations(elements, r))
    return [list(comb) for comb in all_combinations]