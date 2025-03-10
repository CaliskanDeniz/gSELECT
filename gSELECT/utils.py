import os
from itertools import combinations


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