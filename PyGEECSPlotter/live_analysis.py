import os
import time
from threading import Event
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import filelock
import matplotlib.pyplot as plt
import traceback
import re

from PyGEECSPlotter.scan_data_analysis import ScanDataAnalyzer
import PyGEECSPlotter.navigation_utils as nav_utils

class DirectoryWatcherSFile(FileSystemEventHandler):
    def __init__(self, experiment_dir, 
                 analyzers, 
                 search_replace_filename=None,
                 column_math_filename=None):
        """
        A class that watches over a specified directory and looks for scan info files being created
        On creation of a scan info file, the class runs all of the analyzers associated with the class.
        It looks for scan info files in the current day's analysis folder

        Parameters:
        ---------
            experiment_dir (str | Path): directory containing the year folders (e.g. Y2025) for a given experiment
            analyzers (list[Analyzer]): ordered iterable of analyzers to use. The analysis order is given by 
                the list order
            search_replace_filename (str | Path): filename for search and replace columns
            column_math_filename (str | Path): filename for column math
        """
        self.experiment_dir = experiment_dir
        self.top_dir = nav_utils.get_todays_top_dir(experiment_dir)
        self.analysis_dir = os.path.join(self.top_dir, 'analysis')
        self.observer = Observer()
        self.stop_event = Event()
        self.active = False
        
        # Analysis 
        self.search_replace_filename = search_replace_filename
        self.column_math_filename = column_math_filename
        
        # Diagnostics information, containing analyzers, file_ext, analyzer_dict, and display_dict for each diagnostic
        self.analyzers = analyzers

        # Initialize current status and filename tracking
        self.current_status = ""
        self.current_sfile_info_name = None  # Full path to the detected file
        self.current_sfilename = None        # Full path to the transformed filename
        self.current_scan_analyzer = None    # The ScanDataAnalyzer instance for the current file

    def _update_status(self, message):
        """Update the current status and print the message."""
        self.current_status = message
        print(message)

    def start(self):
        """Start the observer."""
        if not os.path.exists(self.analysis_dir):
            self._update_status(f"Directory not found: {self.analysis_dir}")
            self.active = False
            return

        if not self.active:
            self.stop_event.clear()
            self.active = True
            self.observer = Observer()
            self._update_status("Starting directory watcher for analysis files...")
            self.observer.schedule(self, self.analysis_dir, recursive=False)

            try:
                self.observer.start()
            except Exception as e:
                self._update_status(f"Failed to start observer: {e}")
                self.active = False
                return

            try:
                while not self.stop_event.is_set():
                    time.sleep(1)
            except KeyboardInterrupt:
                self.stop()

    def stop(self):
        """Stop the observer."""
        if self.active:
            self._update_status("Stopping directory watcher...")
            self.stop_event.set()
            if self.observer.is_alive():
                self.observer.stop()
                self.observer.join()
            self.active = False

    def on_created(self, event):
        """Handle new file creation events."""
        if not event.is_directory and event.src_path.endswith('info.txt'):
            self.current_sfile_info_name = event.src_path
            self.current_sfilename = self.transform_to_sfilename(self.current_sfile_info_name)
            self.current_scan_analyzer = ScanDataAnalyzer(sfilename=self.current_sfilename)
            self.current_scan_analyzer.load_scan_data()
            self._update_status(f"New file detected: {self.current_sfile_info_name}")
            
            # Run analysis and plotting for each diagnostic in diagnostics_info
            for analyzer in self.analyzers:
                try:
                    self._analyze_and_plot(self.current_sfilename, analyzer)
                except Exception as e:
                    print(f"Exception occured when using analyzer {analyzer.friendly_name}")
                    print("Traceback:")
                    print(traceback.format_exc())
                    continue

    def transform_to_sfilename(self, sfile_info_name):
        """Transform from info filename to sfilename by removing 'info'."""
        if sfile_info_name.endswith('info.txt'):
            return sfile_info_name.replace('info', '')
        return sfile_info_name
    
    def _sfilename_to_scan_data_fname(self, sfilename):
        """Transform '.../s{N}.txt' to '.../scans/Scan{N:03d}/ScanDataScan{N:03d}.txt'."""
        basename = os.path.basename(sfilename)
        match = re.fullmatch(r's(\d+)\.txt', basename)
        if not match:
            return None
        scan_num = int(match.group(1))
        return os.path.join(
            self.top_dir, "scans",
            f"Scan{scan_num:03d}",
            f"ScanDataScan{scan_num:03d}.txt"
        )

    def _analyze_and_plot(self, sfilename, analyzer):
        """Run analysis and plot images for a specific diagnostic."""
        print("-"*50)
        self._update_status(f"Running analysis on {sfilename} for analyzer {analyzer.friendly_name}")
        
        # Unpack diagnostic-specific settings
        file_ext = analyzer.file_ext
        analyzer_dict = analyzer.analyzer_dict
        display_dict = analyzer.display_dict
        diagnostic = analyzer.diagnostic
        
        # Load data files names into ScanDataAnalyzer instance
        self.current_scan_analyzer.add_file_list_to_scan_data(diagnostic, file_ext)
        
        if analyzer_dict.get("use_bg", False):
            bg = self.current_scan_analyzer.get_bg_file_path(diagnostic, file_ext)
        else:
            bg = None
        add_columns_df = self.current_scan_analyzer.analyze_scan(analyzer, 
            bg=bg, 
            display_data=display_dict.get("display_data", False),
            write_columns_to_sfile=False, 
            overwrite_columns=False, 
            analysis_label=analyzer_dict.get("analysis_label", None),
            write_analyzed=analyzer_dict.get("write_analyzed", False),
            write_lineouts=analyzer_dict.get("write_lineouts", False),
            close_displayed=True,
        )

        if add_columns_df is not None:
            print(f"Files analyzed for {analyzer.friendly_name}: {len(add_columns_df)}")
        
        # We use a custom anti-race condition code to update the sfile because the standard ScanDataAnalyzer code
        # does not account for race conditions.
        if analyzer_dict.get('add_columns_to_masterlog', False):
            # file lock to avoid the race condition
            lock_file = sfilename + ".lock"
            lock = filelock.FileLock(lock_file)
            if analyzer_dict.get("analysis_diagnostic", False):
                prefix = analyzer_dict.get("analysis_diagnostic")
            elif analyzer_dict.get('analyze_raw_data', True):
                prefix = diagnostic
            else:
                prefix = None
            with lock:
                merged_sfile = self.current_scan_analyzer.merge_data_frame_to_sfile(add_columns_df, prefix,
                                                                       overwrite_columns=True,
                                                                       analysis_label=analyzer_dict.get("analysis_label", None))
            print("Added columns to masterlog")

        # This code updates the scan_data file inside the scans/ScanXXX folder, 
        # NOT the sfile in the analysis folder
        if analyzer_dict.get('update_scan_data_file', False):
            # file lock to avoid the race condition
            scan_data_fname = self._sfilename_to_scan_data_fname(sfilename)
            lock_file = scan_data_fname + ".lock"
            lock = filelock.FileLock(lock_file)
            if analyzer_dict.get("analysis_diagnostic", False):
                prefix = analyzer_dict.get("analysis_diagnostic")
            elif analyzer_dict.get('analyze_raw_data', True):
                prefix = diagnostic
            else:
                prefix = None
            with lock:
                # Workaround to set the file we are saving to a different location
                # compared to where sfiles are usually saved
                old_sfilename = self.current_scan_analyzer.sfilename
                self.current_scan_analyzer.sfilename = scan_data_fname
                merged_sfile = self.current_scan_analyzer.merge_data_frame_to_sfile(add_columns_df, prefix,
                                                                       overwrite_columns=True,
                                                                       analysis_label=analyzer_dict.get("analysis_label", None))
                # Change the sfilename back to the original one in case we need to save more files
                self.current_scan_analyzer.sfilename = old_sfilename
            print("Added columns to scan data file")
        print("-"*50, "\n")
            
class DirectoryWatcherSFileTester(DirectoryWatcherSFile):
    """
    Testing class, only overrides the top dir
    """
    def __init__(self, top_dir,
                 analyzers,  # Dictionary containing analyzers and parameters for each diagnostic
                 search_replace_filename=None,
                 column_math_filename=None):
        super().__init__(top_dir, analyzers, search_replace_filename, column_math_filename)
        self.top_dir = top_dir
        self.analysis_dir = os.path.join(self.top_dir, "analysis")

class SFileReanalyzer(DirectoryWatcherSFile):
    """
    Re-analyzes sfiles in a certain directory. This class is subclassed from
    DirectoryWatcherSFile to use the methods
    """
    def __init__(self, top_dir, analyzers, search_replace_filename=None,
                 column_math_filename=None):
        """
        Parameters:
        ---------
            top_dir (str | Path): top_dir to analyzes files from
            analyzers (Iterable[Analyzer]): ordered iterable of analyzers to use. 
                The analysis order is given by the iterable order
            search_replace_filename (str | Path): filename for search and replace columns
            column_math_filename (str | Path): filename for column math
        """
        super().__init__(top_dir, analyzers, search_replace_filename, column_math_filename)
        # Replace top_dir and analysis dir
        self.top_dir = top_dir
        self.analysis_dir = os.path.join(self.top_dir, 'analysis')
        # Make sure people don't use this watcher as a directory observer
        self.observer = None
        self.stop_event = None

    def start(self):
        raise NotImplementedError("SFileReanalyzer class is not a watcher! "
                                  "Use DirectoryWatcherSFile instead")
    
    def on_created(self, event):
        raise NotImplementedError("SFileReanalyzer class is not a watcher! "
                                  "Use DirectoryWatcherSFile instead")
    
    def stop(self):
        raise NotImplementedError("SFileReanalyzer class is not a watcher! "
                                  "Use DirectoryWatcherSFile instead")
    
    def analyze_all_scans(self, start_scan=0, end_scan=1000):
        """
        Analyzes all scans inside the current analysis folder
        """
        sfile_list = nav_utils.generate_sfilename_list_from_scans_dir(self.top_dir, start_scan=start_scan, end_scan=end_scan)
        print(sfile_list)
        for sfile in sfile_list:
            self.current_scan_analyzer = ScanDataAnalyzer(sfilename=sfile)
            self.current_scan_analyzer.load_scan_data()
            print("-"*50)
            print(f"Analyzing file {sfile}")
            print("-"*50)
            for analyzer in self.analyzers:
                try:
                    self._analyze_and_plot(sfile, analyzer)
                except Exception as e:
                    print(f"Exception occured when using analyzer {analyzer.friendly_name}")
                    print("Traceback:")
                    print(traceback.format_exc())
                    continue