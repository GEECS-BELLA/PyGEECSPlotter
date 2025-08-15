# PyGEECS navigation functions
# Author: Alex Picksley
# Version 0.4
# Created: 2023-02-26
# Last Modified: 2025-07-30

import numpy as np
import os, sys
import re
import glob
import json
import datetime
from pathlib import Path
import platform
import subprocess

def _format_date(year, month, day):
    datetime_object = datetime.datetime.strptime('%d' % month, '%m')
    month_name = datetime_object.strftime('%b')
    return '%02d-%s' % (month, month_name)

def get_top_dir(experiment_dir, year, month, day, print_data=False):
    formatted_month = _format_date(year, month, day)
    top_dir = os.path.join(experiment_dir, 'Y%d' %year, '%s' %formatted_month ,'%d_%02d%02d' %(year-2000, month, day))
    
    if print_data:
        print('Top Dir                  : %s' %top_dir)
        
    return top_dir
    
def get_top_dir_from_sfilename(sfilename, print_data=False):
    date_part = re.search(r'(\d{2}_\d{4})', sfilename)
    
    if not date_part:
        raise ValueError("Date part not found in the filename.")
    
    date_part = date_part.group(1)
    
    # Parse the date part to get year, month, day
    yy_mmdd = datetime.datetime.strptime(date_part, '%y_%m%d')
    
    year = yy_mmdd.year
    month = yy_mmdd.month
    day = yy_mmdd.day
    
    # Get the top directory up to 'analysis'
    path_parts = sfilename.split(os.sep)
    try:
        analysis_index = path_parts.index('analysis')
    except ValueError:
        raise ValueError("'analysis' directory not found in the file path.")
    
    top_dir = os.sep.join(path_parts[:analysis_index])

    if print_data:
        print('Top Dir                  : %s' %top_dir)
    
    return top_dir, year, month, day
    
def get_sfilename_from_top_dir(top_dir, scan, print_data=False):
    sfilename = os.path.join(top_dir, 'analysis', 's%d.txt' %scan)
    if print_data:
        print('sfilename                  : %s' %sfilename)
        
    return sfilename

def generate_sfilename_list_from_scans_dir(top_dir):
    scans_list = glob.glob( os.path.join(top_dir, 'scans', 'Scan*') )
    scans = [os.path.basename(scans_list[i]) for i in range(len(scans_list))]
    scan_numbers = [int(re.search(r'\d+', s).group()) for s in scans]
    sfilename_list = [os.path.join(top_dir, 'analysis', f's{scan}.txt') for scan in scan_numbers]
    return sfilename_list

def get_todays_top_dir(experiment_dir):
    # Get today's date
    today = pd.date_range(start=datetime.date.today(), periods=1).strftime('%Y%m%d').tolist()[0]
    
    # Extract year, month, and day
    year = int(today[0:4])
    month = int(today[4:6])
    day = int(today[6:8])
    
    # Get month name abbreviation
    formatted_month = _format_date(year, month, day)
    
    # Construct the directory path
    top_dir = os.path.join(
        experiment_dir, 
        f'Y{year}', 
        f'{formatted_month}', 
        f'{year - 2000}_{month:02d}{day:02d}'
    )
    
    return top_dir

def get_analysis_dir(top_dir, scan, save_label=None, make_dir=False, print_data=False):
    if save_label is None:
        analysis_dir = os.path.join(top_dir, 'analysis', 'Scan%03d' %scan)
    else:
        analysis_dir = os.path.join(top_dir, 'analysis', save_label, 'Scan%03d' %scan)
    if make_dir and not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)
    if print_data:
        print('Analysys Dir             : %s' %analysis_dir)
    return analysis_dir

def get_analysis_diagnostic_path(analysis_dir, analysis_diagnostic, scan, shot_num, file_ext='.txt'):
    # Construct the basename with the specified format and file extension
    basename = f'Scan{scan:03d}_{analysis_diagnostic}_{shot_num:03d}{file_ext}'
    
    # Construct the full save path
    save_path = os.path.join(analysis_dir, analysis_diagnostic, basename)
    
    # Ensure the directory exists
    dir_path = os.path.join(analysis_dir, analysis_diagnostic)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    return save_path
    
def get_scan_info_text(top_dir, scan, print_data=False):
    with open(os.path.join(top_dir, 'scans', 'Scan%03d' %scan, 'ScanInfoScan%03d.ini' %scan)) as f:
        lines = f.readlines()
    scan_info_text = lines[2].split('"')[1]
    
    if print_data:
        print('Scan Information  : %s' %scan_info_text)
    return scan_info_text

    
def scan_parameter_with_alias(scan_parameter, scan_data):
    if scan_parameter in scan_data.columns:
        return scan_parameter
    elif len([i for i in scan_data.columns if '%s Alias:' %scan_parameter in i]) == 1:
        return [i for i in scan_data.columns if '%s Alias:' %scan_parameter in i][0]
    else:
        print('Could not find scan_parameter %s' %scan_parameter)
        return scan_parameter
    
def get_param_list_no_alias(param_list):
    param_list_no_alias = []
    for param in param_list:
        param_list_no_alias.append( param.split(' Alias:')[0] )
    return param_list_no_alias

def get_param_list_alias(param_list):
    param_list_alias = []
    for param in param_list:
        param_list_alias.append( param.split(' Alias:')[-1] )
    return param_list_alias

def get_parameter_no_alias(parameter):
    return parameter.split(' Alias:')[0]

def get_parameter_alias(parameter):
    return parameter.split(' Alias:')[-1]

def get_scan_parameter(top_dir, sfile_data):
    scan = sfile_data['scan'][0]
    file_path = os.path.join(top_dir, 'scans', 'Scan%03d' % scan, 'ScanInfoScan%03d.ini' % scan)
    with open(file_path, 'r') as f:
        lines = f.readlines()
    scan_parameter_full = scan_parameter_with_alias(lines[3].split('"')[1], sfile_data)
    scan_parameter = get_parameter_alias(scan_parameter_full)
    return scan_parameter, scan
    
def open_directory_in_explorer(path):
    """
    Opens the given directory in the system's file explorer.

    Parameters:
    - path (str): Path to the directory.
    """
    if not os.path.isdir(path):
        raise ValueError(f"The path '{path}' is not a valid directory.")

    system = platform.system()
    if system == "Windows":
        subprocess.run(["explorer", path])
    elif system == "Darwin":  # macOS
        subprocess.run(["open", path])
    else:  # Assume Linux
        subprocess.run(["xdg-open", path])