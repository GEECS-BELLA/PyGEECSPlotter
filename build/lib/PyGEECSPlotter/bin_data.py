import numpy as np
import pandas as pd

def bin_scan_data(scan_data, scan_parameter, method='zscore', zscore_thresh=1, rounding_factor=1):
    """
    Bins the scan_data based on the specified method.

    Parameters:
    - scan_data (pd.DataFrame): The input data containing the scan parameter.
    - scan_parameter (str): The column name in scan_data to be binned.
    - method (str): The binning method to use ('zscore' or 'rounding').

    Returns:
    - scan_data (pd.DataFrame): The DataFrame with an added 'Bin Number' column.
    - bins (list or np.ndarray): The bin edges or centers used.
    """
    data_values = scan_data[scan_parameter].values

    if method == 'zscore':
        # Use zscore_binning function
        bins_list = zscore_binning(data_values, zscore_thresh)
        
        # Assign bin numbers to data points
        bin_numbers = np.zeros(len(data_values), dtype=int)
        sorted_indices = np.argsort(data_values)
        idx_in_sorted = np.argsort(sorted_indices)  # To map back to original order
        start_idx = 0
        for bin_num, b in enumerate(bins_list, start=1):
            end_idx = start_idx + len(b)
            original_indices = sorted_indices[start_idx:end_idx]
            bin_numbers[original_indices] = bin_num
            start_idx = end_idx

        scan_data['temp Bin number'] = bin_numbers
        return scan_data

    elif method == 'rounding':
        # Perform rounding
        rounded_xvals = np.round(data_values / rounding_factor) * rounding_factor
        tmp_bins = np.unique(rounded_xvals)
        
        # Assign bin numbers
        bin_numbers = np.zeros(len(data_values), dtype=int)
        for i in range(len(data_values)):
            bin_index = np.argmin(np.abs(tmp_bins - data_values[i]))
            bin_numbers[i] = bin_index + 1  # Bins start from 1

        scan_data['temp Bin number'] = bin_numbers
        return scan_data
    
    elif method == 'unique':
        tmp_bins = np.unique(scan_data[scan_parameter])
        
        # Assign bin numbers
        bin_numbers = np.zeros(len(data_values), dtype=int)
        for i in range(len(data_values)):
            bin_index = np.argmin(np.abs(tmp_bins - data_values[i]))
            bin_numbers[i] = bin_index + 1  # Bins start from 1

        scan_data['temp Bin number'] = bin_numbers
        return scan_data

    else:
        return scan_data


def zscore_binning(data, z_threshold=3):
    """
    Splits the input array into bins by detecting significant gaps using the Z-score method.

    Parameters:
    - data (array-like): The input array of numerical values.
    - z_threshold (float): The Z-score threshold to identify significant gaps (default is 3).

    Returns:
    - bins (list of numpy arrays): A list containing the binned data as numpy arrays.
    """
    # Step 1: Sort the array
    sorted_data = np.sort(data)

    # Step 2: Compute differences between consecutive elements
    differences = np.diff(sorted_data)

    # Step 3: Compute mean and standard deviation of differences
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)
    if std_diff == 0:
        # All differences are the same, no significant gaps
        return [sorted_data]

    # Step 4: Calculate Z-scores for the differences
    z_scores = (differences - mean_diff) / std_diff

    # Step 5: Identify indices of significant gaps
    gap_indices = np.where(z_scores > z_threshold)[0]

    # Step 6: Split the array into bins based on significant gaps
    bins = []
    start_idx = 0
    for idx in gap_indices:
        end_idx = idx + 1  # Include the element at idx
        bins.append(sorted_data[start_idx:end_idx])
        start_idx = end_idx
    # Add the last bin
    bins.append(sorted_data[start_idx:])

    return bins



def get_bin_values(scan_data, scan_parameter):
    bin_idcs = np.unique(scan_data['temp Bin number'])
    bin_values = []
    shots_per_bin = []

    for bin_idx in bin_idcs:
        idcs = np.where(scan_data['temp Bin number'] == bin_idx)[0]
        mean_value = np.nanmean(scan_data[scan_parameter][idcs])
        if np.isnan(mean_value):
            mean_value = 0.0

        bin_values.append(mean_value)
        shots_per_bin.append(len(idcs))

    return bin_idcs, bin_values, shots_per_bin
