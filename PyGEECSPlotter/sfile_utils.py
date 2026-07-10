# Utilities for working with sfiles
# Author: Raymond Li
# Version 0.4
# Created: 2026-05-29
# Last Modified: 2026-05-29

import os
import pandas as pd

from PyGEECSPlotter.navigation_utils import generate_sfilename_list_from_scans_dir, get_top_dir_from_sfilename


def update_masterlog(masterlog_data, scan_data):
    """
    Updates an sfile with another sfile. All the columns of the masterlog will be overwritten
    with the columns from the sfile for the scan and shotnumbers of the new sfile

    Parameters:
    --------
        masterlog_data: masterlog scan_data, will be updated
        scan_data: scan_data to update with

    Returns:
    --------
        updated_masterlog: masterlog with updated data
    """
    merged = masterlog_data.merge(
        scan_data,
        on=['scan', 'Shotnumber'],
        how='outer',
        suffixes=('', '_update')
    )

    for col in scan_data.columns:
        if col not in ['scan', 'Shotnumber'] and col in masterlog_data.columns:
            update_col = f'{col}_update'
            merged[col] = merged[update_col].fillna(merged[col])
            merged = merged.drop(columns=[update_col])

    return merged


def update_masterlog_with_sfiles(top_dir, columns="all", masterlog_name=None):
    """
    Updates the masterlog with all the columns from the sfiles for a given day.

    If a masterlog doesn't already exist, creates a new masterlog with the columns specified

    Parameters:
    ------------
        top_dir: top directory for the day
        columns: if "all", update all the columns. The masterlog will have a union of
            all the columns in each sfile. If not "all", this parameter should be an iterable
            of column names to update
        masterlog_name: alternative name for masterlog file. If None, will default to
            {year}_{month}{day}masterlog-t.txt

    Returns:
    ----------
        masterlog_data: scan_data pandas dataframe for the masterlog
    """
    if columns != "all":
        columns = list(columns)
        if 'scan' not in columns:
            columns.append('scan')
        if 'Shotnumber' not in columns:
            columns.append('Shotnumber')

    all_sfiles = generate_sfilename_list_from_scans_dir(top_dir)
    if not all_sfiles:
        raise FileNotFoundError(f"No sfiles found under {os.path.join(top_dir, 'scans')}")
    _, year, month, day = get_top_dir_from_sfilename(all_sfiles[0])
    if masterlog_name is None:
        masterlog_name = f"{str(year)[-2:]}_{month:02d}{day:02d}masterlog-t.txt"

    masterlog_path = os.path.join(top_dir, "analysis", masterlog_name)

    masterlog = pd.DataFrame()
    masterlog_exists_flag = False
    if os.path.exists(masterlog_path):
        try:
            masterlog = pd.read_csv(masterlog_path, sep='\t')
            print(f"Found existing masterlog with {len(masterlog)} rows and {len(masterlog.columns)} columns")
            masterlog_exists_flag = True
        except Exception as e:
            print(f"Warning: Could not read existing masterlog {masterlog_path}: {e}")

    frames = []
    for sfilename in all_sfiles:
        print(f"Merging {sfilename}")
        scan_data = pd.read_csv(sfilename, sep='\t')
        frames.append(scan_data if columns == "all" else scan_data[columns])
    temp_masterlog = pd.concat(frames, ignore_index=True)

    masterlog = update_masterlog(masterlog, temp_masterlog) if masterlog_exists_flag else temp_masterlog

    os.makedirs(os.path.dirname(masterlog_path), exist_ok=True)

    try:
        masterlog.to_csv(masterlog_path, sep='\t', index=False)
        print(f"Masterlog written to: {masterlog_path}")
    except Exception as e:
        print(f"Error writing masterlog to {masterlog_path}: {e}")

    return masterlog
