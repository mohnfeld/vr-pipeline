from segment_functions import *
from ecg_functions import *
import os
import pandas as pd
import numpy as np
#from biosppy.signals.tools import filter_signal
#import biosppy.signals.ecg as ecg
#from shapely.geometry import Polygon, LineString
import glob

def find_latest_ecg_directory(base_dir, participant_id, session_id):
    """Find the most recent ECG data directory for a participant"""
    # ecg_path_pattern = os.path.join(base_dir, str(participant_id),
    #                                  'virtual_reality', 'BBT-BIO-AAB*')
    ecg_path_pattern= os.path.join(base_dir, str(participant_id), f'S{session_id:03d}', 'BBT-BIO-AAB*')
    #ecg_path_pattern = os.path.join(base_dir, str(participant_id))
    ecg_dirs = glob.glob(ecg_path_pattern)
    if not ecg_dirs:
        return None
    return max(ecg_dirs)

def main(segment_length=90, step_size=1, participant_sessions: list =None, filename_prefix="segment_data"):
    # Setup parameters
    #base_dir = r"C:\Users\lal\Documents\tez\virtualreality_data_version1"
    #base_dir = r"C:\Users\lal\Documents\tez\virtualreality_data_version1\newpart"
    vr_base_dir= os.path.join(os.getcwd(), "data", "vr")
    base_dir = os.path.join(os.getcwd(), "data")
    #corrected_r_peaks_file_pattern = os.path.join(base_dir, "ecg", f"{participant_id}_{version}_r_peaks_edited_*.npy")
    # participant_sessions = [
    #     (845, 1),
    #     (478, 2),
    #     (578, 2),
    #     (352, 1),
    #     (401, 1),
    #     (2, 1),
    #     (382, 2),
    #     (490, 1),
    #     (6463, 1),
    #     (240, 1),
    #     (497,1),
    #     (851, 1)
    # ]

    # participant_sessions=[
    #     ('EK_207',1), ('EK_207',2), ('EK_240',1), ('EK_240',2), ('EK_260',1), ('EK_352',1), ('EK_352', 2),
    #     ('EK_382',1), ('EK_382',2),  ('EK_401',1), ('EK_401',2), ('EK_478',1), ('EK_478',2), ('EK_490',1), ('EK_490',2),
    #     ('EK_497',1), ('EK_497',2), ('EK_578',1), ('EK_578',2), ('EK_648', 1), ('EK_711',1), ('EK_845',1), ('EK_845',2),
    #     ('EK_851',1), ('EK_851',2), ('EK_855',1), ('EK_882',1), ('EK_882',2), ('EK_945',1), ('EK_951',2), ('EK_993',1)
    # ]
    
    participant_sessions = participant_sessions

    trials = [7, 11, 15]

    def get_trial_coords(trial_num, version=1):
        """Return the appropriate coordinates for each trial"""
        coords = {
    7: {
        'center_point': np.array((-6.14, 5.05)),
        'corner_points': np.array([(-8.12, 8.87), (-4.30, 8.87), (-4.30, 0.98), (-8.12, 0.98), (-8.12, 8.87)]),
    },
    11: {
        'center_point': np.array((-6.14, 5.05)),
        'corner_points': [
            #room boundary
            np.array([(-8.12, 8.87), (-4.30, 8.87), (-4.30, 0.98), (-8.12, 0.98), (-8.12, 8.87)]),
            np.array([(-6.00, 1.73), (-6.60, 1.73), (-6.60, 2.89), (-7.20, 2.89), 
                                 (-7.20, 4.66), (-7.80, 4.66), (-7.80, 7.00), (-4.80, 7.00), 
                                 (-4.80, 4.66), (-5.40, 4.66), (-5.40, 2.89), (-6.00, 2.89)])]
    },

    15: {
        1: {
        'center_point': np.array((-6.14, 5.05)),
        'corner_points': [
            # Main room boundary
            np.array([(-8.12, 8.87), (-4.30, 8.87), (-4.30, 0.98), (-8.12, 0.98), (-8.12, 8.87)]),
            # wall1 down wall
            np.array([
            (-7.36, 2.60),  # right up
            (-7.36, 2.30),  # right down
            (-7.96, 2.30),  # left down
            (-7.96, 2.60)   # left up
        ]),
            # wall2
            np.array([
            (-6.72, 4.90),  # right up
            (-6.72, 4.31),  # right down
            (-7.02, 4.31),  # left down
            (-7.02, 4.90)   # left up
        ]),
            # wall3 upper wall
            np.array([
            (-4.30, 7.81),  # right up
            (-4.30, 7.60),  # right down
            (-6.90, 7.60),  # left down
            (-6.90, 7.81)   # left up
        ]),
            # wall4 t shape
            np.array([
            (-7.06, 5.95),
            (-7.25, 5.95),
            (-7.25, 5.16),
            (-8.02, 5.16),
            (-8.02, 4.95),
            (-7.25, 4.95),
            (-7.25, 4.31),
            (-7.06, 4.31),
            (-7.06, 4.95),
            (-6.04, 4.95),
            (-6.04, 5.16),
            (-7.06, 5.16),
            (-7.06, 5.95)
        ])
        ]},
        2: {
        'center_point': np.array((-5.14, 9.05)),
        'corner_points': [
    # Main room boundary (same)
            np.array([(-8.12, 8.87), (-4.30, 8.87), (-4.30, 0.98), (-8.12, 0.98), (-8.12, 8.87)]),
            
            # Mirrored wall1 (original right → new left)
            np.array([
                (-4.52, 2.60),   # left up (original -7.36 → -4.52)
                (-4.52, 2.30),   # left down
                (-3.92, 2.30),   # right down (original -7.96 → -3.92)
                (-3.92, 2.60)    # right up
            ]),
            
            # Mirrored wall2 
            np.array([
                (-5.36, 4.90),  # left up (original -6.72 → -5.36)
                (-5.36, 4.31),  # left down
                (-5.06, 4.31),  # right down (original -7.02 → -5.06)
                (-5.06, 4.90)   # right up
            ]),
            
            # Mirrored wall3 
            np.array([
                (-4.30, 7.81),  # right up (same boundary edge)
                (-4.30, 7.60),  # right down
                (-5.38, 7.60),  # left down (original -6.90 → -5.38)
                (-5.38, 7.81)   # left up
            ]),
            
            # Mirrored wall4 (T-shape flipped)
            np.array([
                (-5.22, 5.95),  # Original -7.06 → -5.22
                (-5.03, 5.95),  # Original -7.25 → -5.03
                (-5.03, 5.16),  # Original -7.25 → -5.03
                (-4.26, 5.16),  # Original -8.02 → -4.26
                (-4.26, 4.95),  # 
                (-5.03, 4.95),  # 
                (-5.03, 4.31),  # 
                (-5.22, 4.31),  # 
                (-5.22, 4.95),  # 
                (-6.24, 4.95),  # Original -6.04 → -6.24
                (-6.24, 5.16),  # 
                (-5.22, 5.16),  # 
                (-5.22, 5.95)
            ])
        ]}
    }
}

        if trial_num==15:
            return coords.get(trial_num, {}).get(version, coords[trial_num][1])
        return coords.get(trial_num)

    def get_segment(data, start_time, end_time, timestamps):
        """Extract segment of data between start and end times"""
        mask = (timestamps >= start_time) & (timestamps <= end_time)
        return data.iloc[mask] if isinstance(data, pd.DataFrame) else data[mask]

    def process_single_trial(movement_path, ecg_dir, trial_num, version=1):
        coords = get_trial_coords(trial_num, version)
        if coords is None:
            print(f"No coordinates defined for trial {trial_num}")
            return None
            
        try:
            # Load movement data
            if not os.path.exists(movement_path):
                print(f"Movement file not found: {movement_path}")
                return None
                
            movement_data = pd.read_csv(movement_path)
            if movement_data.empty:
                print(f"Empty movement data file: {movement_path}")
                return None

            # Load ECG data if available
            ecg_data = None
            rmssd = None
            signal_al = None
            heartrate_timestamps_aligned = None

            if ecg_dir and os.path.exists(ecg_dir):
                ecg_path = os.path.join(ecg_dir, 'ExG [1].csv')
                utc_path = os.path.join(ecg_dir, 'UTC.csv')
                
                if os.path.exists(ecg_path) and os.path.exists(utc_path):
                    try:
                        heartrate_data = pd.read_csv(ecg_path)
                        utc_df = pd.read_csv(utc_path)
                        
                        if not heartrate_data.empty and not utc_df.empty:
                            timestamp_diff = (utc_df['utc_timestamp']-utc_df['steady_timestamp']).astype(int).to_numpy()[0]
                            heartrate_timestamps = ((heartrate_data['steady_timestamp'] + timestamp_diff)/1e6).astype(int).to_numpy()
                            
                            # Process timestamps for movement data
                            timestamps = pd.to_datetime(movement_data['timestamp']).apply(lambda x: x.timestamp()).astype(int).to_numpy()
                            movement_min = np.min(timestamps)
                            movement_max = np.max(timestamps)

                            # Align heart rate data to movement data range
                            mask = (heartrate_timestamps >= movement_min) & (heartrate_timestamps <= movement_max)
                            heartrate_data_aligned = heartrate_data[mask]
                            heartrate_timestamps_aligned = heartrate_timestamps[mask]
                            signal_al = heartrate_data_aligned['ExG [1]-ch1']

                    except Exception as e:
                        print(f"Error processing ECG data: {str(e)}")
                        signal_al = None
                        heartrate_timestamps_aligned = None

            # Process timestamps
            timestamps = pd.to_datetime(movement_data['timestamp']).apply(lambda x: x.timestamp()).astype(int).to_numpy()
            
            if len(timestamps) == 0:
                print(f"No valid timestamps in file: {movement_path}")
                return None

            # Setup time windows with customizable parameters
            start_time = np.floor(np.min(timestamps))
            if trial_num == 11:  #you can change the start time of trial 11 (elevated platform) as you like 
                start_time += 30
            end_time = start_time + 90 #elevated platform is 130 seconds in total

            
                
            segment_starts = np.arange(start_time, end_time-segment_length+1, step=step_size)
            segment_ends = segment_starts + segment_length

            # Initialize arrays for metrics
            segment_mean_center_dist = np.zeros(len(segment_starts))
            segment_mean_speed = np.zeros(len(segment_starts))
            segment_mean_edge_dist = np.zeros(len(segment_starts))
            segment_mean_acc = np.zeros(len(segment_starts))
            segment_stops_count = np.zeros(len(segment_starts))
            segment_stops_duration = np.zeros(len(segment_starts))
            segment_max_distance = np.zeros(len(segment_starts))
            segment_mean_area_covered = np.zeros(len(segment_starts))
            segment_rmssd = np.zeros(len(segment_starts))
            segment_hr = np.zeros(len(segment_starts))

            # Process segments using trial-specific coordinates
            valid_segments = 0
            for idx, (start, end) in enumerate(zip(segment_starts, segment_ends)):
                movement_segment = get_segment(movement_data, start, end, timestamps)
                
                if signal_al is not None and heartrate_timestamps_aligned is not None:
                    corrected_r_peaks_file = corrected_r_peaks_file_pattern.format(participant_id=participant_id)
                    corrected_r_peaks_files = glob.glob(corrected_r_peaks_file)
                    #heart_segment = get_segment(signal_al, start, end, heartrate_timestamps_aligned)
                    #segment_rmssd[idx] = get_segment_rmssd(heart_segment, 250)
                    #segment_hr[idx] = get_segment_heart_rate(heart_segment, 250)
                    if corrected_r_peaks_files:
                       corrected_r_peaks_file = corrected_r_peaks_files[0]
                       corrected_r_peaks = np.load(corrected_r_peaks_file)
                       #hr_segment = get_segment(ecg_cleaned, start, end, heartrate_timestamps_aligned)
                       start_index = np.searchsorted(heartrate_timestamps_aligned, start)
                       end_index = np.searchsorted(heartrate_timestamps_aligned, end)
                       segment_r_peak_indices = corrected_r_peaks[(corrected_r_peaks >= start_index) & (corrected_r_peaks <= end_index)]
                       rri, _ = peaks_to_rri(peaks=segment_r_peak_indices, sampling_rate=256, interpolate=True, filter_outliers=True)
                       rmssd=calc_rmssd(rri)
                       segment_rmssd[idx] = rmssd
                       hr=get_segment_heart_rate(rri)
                       segment_hr[idx] = hr
                       #segment_rmssd[idx]=np.nan
                       #segment_hr[idx]=np.nan
                else:
                    segment_rmssd[idx] = np.nan
                    segment_hr[idx] = np.nan
                
                if len(movement_segment) > 0:
                    segment_mean_center_dist[idx] = get_segment_mean_center_dist(movement_segment, coords['center_point'])
                    segment_mean_edge_dist[idx] = get_segment_mean_edge_dist(movement_segment, coords['corner_points'])
                    segment_mean_speed[idx] = get_segment_speed(movement_segment)
                    segment_mean_acc[idx] = get_segment_acceleration(movement_segment)
                    segment_stops_count[idx] = get_segment_stops_count(movement_segment)
                    segment_stops_duration[idx] = get_segment_stops_duration(movement_segment)
                    segment_max_distance[idx] = get_max_distance(movement_segment)
                    segment_mean_area_covered[idx] = get_segment_mean_area_covered(movement_segment, coords['corner_points'])
                    valid_segments += 1
                else:
                    segment_mean_center_dist[idx] = np.nan
                    segment_mean_edge_dist[idx] = np.nan
                    segment_mean_speed[idx] = np.nan

            if valid_segments == 0:
                print(f"No valid segments found in file: {movement_path}")
                return None

            return {
                'center_dist': segment_mean_center_dist[~np.isnan(segment_mean_center_dist)],
                'edge_dist': segment_mean_edge_dist[~np.isnan(segment_mean_edge_dist)],
                'speed': segment_mean_speed[~np.isnan(segment_mean_speed)],
                'acceleration': segment_mean_acc[~np.isnan(segment_mean_acc)],
                'stops_count': segment_stops_count,
                'stops_duration': segment_stops_duration,
                'max_distance': segment_max_distance,
                'area_covered': segment_mean_area_covered[~np.isnan(segment_mean_area_covered)],
                'rmssd': segment_rmssd,
                'hr': segment_hr,
                'timestamps': segment_starts
            }

        except Exception as e:
            print(f"Error processing file {movement_path}: {str(e)}")
            return None

    # Process all data and create segment DataFrame
    segment_data_list = []
    
    for participant_info in participant_sessions:
        participant_id = participant_info[0]
        session_id = participant_info[1]
        version = participant_info[2] if len(participant_info) > 2 else 1


        ecg_dir = find_latest_ecg_directory(vr_base_dir, participant_id, session_id)
        corrected_r_peaks_file_pattern = os.path.join(base_dir, "ecg", f"{participant_id}_{session_id}_{version}_r_peaks_edited_*.npy")
        
        for trial in trials:
            # movement_path = os.path.join(base_dir, 'vr', str(participant_id), 'virtual_reality', 
            #                            f'S{session_id:03d}', 'trackers_rotated', 
            #                            f'camera_movement_T{trial:03d}.csv')
            
            movement_path = os.path.join(base_dir, str(participant_id), 
                                       f'S{session_id:03d}', 'trackers_rotated', 
                                       f'camera_movement_T{trial:03d}.csv')
            
            if os.path.exists(movement_path):
                trial_data = process_single_trial(movement_path, ecg_dir, trial, version=version)
                if trial_data is not None:
                    for seg_idx, (center_d, edge_d, speed, acc, stops_count, stops_duration, 
                                max_distance, area_covered, rmssd, hr) in enumerate(zip(
                                    trial_data['center_dist'],
                                    trial_data['edge_dist'],
                                    trial_data['speed'],
                                    trial_data['acceleration'],
                                    trial_data['stops_count'],
                                    trial_data['stops_duration'],
                                    trial_data['max_distance'],
                                    trial_data['area_covered'],
                                    trial_data['rmssd'],
                                    trial_data['hr'])):
                        segment_data_list.append({
                            'participant_id': participant_id,
                            'session': session_id,
                            'trial': trial,
                            'version': version,
                            'segment': seg_idx,
                            'center_dist': center_d,
                            'edge_dist': edge_d,
                            'speed': speed,
                            'acceleration': acc,
                            'stops_count': stops_count,
                            'stops_duration': stops_duration,
                            'max_distance': max_distance,
                            'area_covered': area_covered,
                            'rmssd': rmssd,
                            'hr': hr
                        })
    
    # Create DataFrame and save

    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True) 
    segment_data = pd.DataFrame(segment_data_list)
    file_path = os.path.join(results_dir, f'{filename_prefix}_{segment_length}s_{step_size}s.csv')
    segment_data.to_csv(file_path, index=False)
    
    #segment_data.to_csv(f'segment_data_bea_{segment_length}s_{step_size}s.csv', index=False)
    
    return segment_data

if __name__ == "__main__":
    segment_data = main(segment_length=90, step_size=1)  