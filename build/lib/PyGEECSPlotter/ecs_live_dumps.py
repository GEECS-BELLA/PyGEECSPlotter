# GEECS Plotter for extracting from ECS Live Dumps
# Author: Alex Picksley
# Version 0.3
# Created: 2024-02-26
# Last Modified: 2025-08-14

import numpy as np
import os
import pandas as pd

from PyGEECSPlotter.navigation_utils import get_top_dir_from_sfilename


def add_ecs_live_dump_value_to_sfile(sfilename, top_dir, device, parameter, print_data=True):
    sfile_data = pd.read_csv(sfilename, sep='\t')
    
    # Check if 'scan' column exists
    if 'scan' not in sfile_data.columns:
        return np.nan, np.nan
    
    scans_in_sfile = np.unique(sfile_data['scan'])
    parameter_scan_values = np.zeros(len(scans_in_sfile))

    for si in range(len(scans_in_sfile)):
        scan = scans_in_sfile[si]
        ecs_live_path = os.path.join(top_dir, 'ECS Live dumps', 'Scan%d.txt' %scan)
        if os.path.exists(ecs_live_path):
            parameter_scan_values[si] = get_ecs_live_dumps_parameter_value(ecs_live_path, device, parameter)

    new_column_from_ecs = np.zeros(len(sfile_data))
    for i in range(len(sfile_data)):
        idx = np.where(scans_in_sfile == sfile_data['scan'][i])[0][0]
        new_column_from_ecs[i] = parameter_scan_values[idx]

    sfile_data['%s %s' %(device, parameter)] = new_column_from_ecs
    sfile_data.to_csv(sfilename, index=False, sep='\t')
    if print_data:
        print('Columns added to %s' %sfilename)
        
    return scans_in_sfile, parameter_scan_values


def get_ecs_live_dumps_parameter_value(file_path, device_name, parameter_name, print_data=False):
    with open(file_path, 'r') as file:
        content = file.read()
    
    devices = content.split("[Device")
    devices.pop(0)
    
    for device in devices:
        if f'Device Name = "{device_name}"' in device:
            lines = device.split("\n")
            for line in lines:
                if line.strip().startswith(f'{parameter_name} = '):
                    value = line.split('=')[1].strip().strip('"')
                    return np.float64(value)
    
    
    if print_data:
        print("Device or parameter not found.")
    return np.nan
    
def get_ecs_live_dumps_parameter_value_from_sfilename(sfilename, device, parameter):
    """
    Reads a scan file and retrieves the parameter value from ECS Live dumps if the scan is valid.

    Parameters:
    - sfilename: str, the path to the scan file
    - device: str, the device name for which the parameter is being retrieved
    - parameter: str, the parameter name to retrieve

    Returns:
    - parameter_scan_value: The value of the parameter, or None if not found.
    """
    top_dir, year, month, day = get_top_dir_from_sfilename(sfilename, print_data=False)
    sfile_data = pd.read_csv(sfilename, sep='\t')
    
    if 'scan' not in sfile_data.columns or len(sfile_data) == 0:
        print('Scan column not found or file is empty.')
        return np.nan, np.nan
    
    scan = sfile_data['scan'][0]
    ecs_live_path = os.path.join(top_dir, 'ECS Live dumps', 'Scan%d.txt' %scan)

    if os.path.exists(ecs_live_path):
        return scan, get_ecs_live_dumps_parameter_value(ecs_live_path, device, parameter)
    
    print(f'Path {ecs_live_path} does not exist.')
    return scan, np.nan

def sort_scan_sheet_by_time(scan_sheet):
    """
    Sort the DataFrame by year, month, day, and scan, and create a new timestamp column.

    Parameters:
    scan_sheet (pd.DataFrame): The DataFrame containing 'year', 'month', 'day', and 'scan' columns.

    Returns:
    pd.DataFrame: The sorted DataFrame with an additional 'timestamp' column.
    """
    
    scan_sheet = scan_sheet.dropna(subset=['scan'])
    scan_sheet = scan_sheet.sort_values(by=['year', 'month', 'day', 'scan'])
    
    # Ensure timestamp is a single string for each row
    scan_sheet['timestamp'] = scan_sheet.apply(
        lambda row: f"{int(row['year'])}-{int(row['month']):02d}-{int(row['day']):02d}_s{int(row['scan']):03d}",
        axis=1
    )

    return scan_sheet