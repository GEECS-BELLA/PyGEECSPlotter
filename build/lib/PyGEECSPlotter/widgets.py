import numpy as np
import os
import re
import glob
import json
import datetime
import pandas as pd
import ipywidgets as widgets
from IPython.display import display

def get_selection_box(experiment_dir, display_widgets=True):
    def extract_number(filename):
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else float('inf')

    def update_month_select(*args):
        month_select.options = [os.path.basename(str) for str in 
                                sorted(glob.glob(os.path.join(experiment_dir, year_select.value, '*-*')))]
        if len(month_select.options) > 0:
            month_select.value = month_select.options[-1]

    def update_date_select(*args):
        if month_select.value is not None:
            year_str = year_select.value
            year_val = int(year_str[1:])-2000
            date_select.options = [os.path.basename(str) for str in 
                                   sorted(glob.glob(os.path.join(experiment_dir, year_select.value,  
                                                                 month_select.value, '%d_*' %year_val)))]
            if len(date_select.options) > 0:
                date_select.value = date_select.options[-1]

    def update_scan_select(*args):
        if date_select.value is not None:
            txt_files_list = sorted(glob.glob(os.path.join(experiment_dir, year_select.value, 
                                                           month_select.value,  date_select.value, 'analysis' ,'*.txt')))
            txt_files_list = [os.path.basename(item) for item in txt_files_list if "info.txt" not in item]
            sorted_list = sorted(txt_files_list, key=extract_number)

            scan_select.options = sorted_list
            if len(scan_select.options) > 0:
                scan_select.value = scan_select.options[-1]

    def update_scan_info_txt_display(*args):
        if month_select.value is not None and date_select.value is not None and scan_select.value is not None:
            tmp_scan = extract_number(scan_select.value)
            if not np.isinf(tmp_scan):
                tmp_dir = os.path.join(experiment_dir, year_select.value, month_select.value, 
                                            date_select.value, 'scans', 'Scan%03d' %tmp_scan)

                if os.path.exists(os.path.join(tmp_dir, 'ScanInfoScan%03d.ini' %tmp_scan)):
                    with open(os.path.join(tmp_dir, 'ScanInfoScan%03d.ini' %tmp_scan)) as f:
                        lines = f.readlines()
                        scan_info_txt_disp.value = lines[2].split('"')[1]

    def update_sfile_display(*args):
        if month_select.value is not None and date_select.value is not None and scan_select.value is not None:
            tmp_dir = os.path.join(experiment_dir, year_select.value, month_select.value, 
                                        date_select.value, 'analysis', scan_select.value)
            sfile_disp.value = tmp_dir


    style = {'description_width': 'initial'}
    exp_dir_disp = widgets.Text(value=experiment_dir, description='Experiment Dir :', style=style, 
                                          layout=widgets.Layout(width='90%'), disabled=True)

    year_opts = [os.path.basename(str) for str in sorted(glob.glob(os.path.join(experiment_dir, 'Y*')))] 

    year_select = widgets.Select(options=year_opts, description='Year:', rows=12, disabled=False, value=year_opts[-1])
    month_select = widgets.Select(description='Month:', rows=12, disabled=False)

    year_select.observe(update_month_select)
    widgets.interact(update_month_select, x = year_select, y = month_select)

    date_select = widgets.Select(description='Date:', rows=12, disabled=False)
    month_select.observe(update_date_select)
    widgets.interact(update_date_select, x = month_select, y = date_select)

    scan_select = widgets.Select(description='Scan:', rows=12, disabled=False)
    date_select.observe(update_scan_select)
    widgets.interact(update_scan_select, x = date_select, y = scan_select)

    selection_box = widgets.Box(children=[year_select, month_select, date_select, scan_select])
    #     selection_box = widgets.Box(children=[year_select])

    scan_info_txt_disp = widgets.Textarea(description='Scan Information :', style=style, 
                                          layout=widgets.Layout(width='90%'), disabled=True)

    sfile_disp = widgets.Textarea(description='sfile            :', style=style, 
                                          layout=widgets.Layout(width='90%'), disabled=True)

    scan_select.observe(update_scan_info_txt_display)
    widgets.interact(update_scan_info_txt_display, x = scan_select, y = scan_info_txt_disp)

    scan_select.observe(update_sfile_display)
    widgets.interact(update_sfile_display, x = scan_select, y = sfile_disp)

    if display_widgets:
        display(widgets.VBox([exp_dir_disp, selection_box, scan_info_txt_disp, sfile_disp]))

    return selection_box, year_select, month_select, date_select, scan_select, sfile_disp
    


def get_txt_files_in_folder(folder_path):
    # Get all .txt files in the folder, including their full paths
    txt_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]
    # Sort the files by modification time (most recent first)
    txt_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return txt_files

def filter_files(txt_files, filter_string):
    # Filter the files based on the filter_string in the basename
    return [f for f in txt_files if filter_string in os.path.basename(f)]

def create_filtered_dropdown_widget(folder_path, default_filter_string=''):
    
    def update_dropdown(change):
        # Get the filter string from the text box
        filter_string = filter_text.value
        # Filter the files based on the filter string
        filtered_files = filter_files(all_txt_files, filter_string)
        # Update the dropdown options to only show basenames but keep full paths
        if filtered_files:
            dropdown.options = [(os.path.basename(f), f) for f in filtered_files]
        else:
            dropdown.options = [('No matching files found', None)]
    
    # Get the list of all .txt files in the folder
    all_txt_files = get_txt_files_in_folder(folder_path)
    
    # Create a dropdown widget, with initial filter applied
    filtered_files = filter_files(all_txt_files, default_filter_string)
    dropdown = widgets.Dropdown(
        options=[(os.path.basename(f), f) for f in filtered_files] if filtered_files else [('No matching files found', None)],
        description='Select File:',
        disabled=False,
        layout=widgets.Layout(width='50%')
    )
    
    # Create a text entry widget for filtering, with the default filter string
    filter_text = widgets.Text(
        value=default_filter_string,
        description='Filter:',
        layout=widgets.Layout(width='50%')
    )
    
    # Link the text entry to the function that updates the dropdown
    filter_text.observe(update_dropdown, names='value')
    
    # Display the filter text box and dropdown menu
    display(filter_text, dropdown)
    return dropdown


