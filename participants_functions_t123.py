#### originals + unsafe safe area covered

from segment_functions import *
from ecg_functions import *
import os
import pandas as pd
import numpy as np
from shapely.geometry import LineString, Polygon
import glob

def find_latest_ecg_directory(base_dir, participant_id, session_id):
    #ecg_path_pattern = os.path.join(base_dir, str(participant_id), 'virtual_reality', 'BBT-BIO-AAB*')
    ecg_path_pattern= os.path.join(base_dir, 'RAW1', str(participant_id), f'S{session_id:03d}', 'BBT-BIO-AAB*')
    ecg_dirs = glob.glob(ecg_path_pattern)
    if not ecg_dirs:
        return None
    return max(ecg_dirs)

def get_trial_coords(trial_num, version=1):
    coords = {
        7: {
            'center_point': np.array((-6.14, 5.05)),
            'corner_points': np.array([(-8.12, 8.87), (-4.30, 8.87), (-4.30, 0.98), (-8.12, 0.98), (-8.12, 8.87)]),
        },
        11: {
            'center_point': np.array((-6.14, 5.05)),
            'corner_points': [
                np.array([(-8.12, 8.87), (-4.30, 8.87), (-4.30, 0.98), (-8.12, 0.98), (-8.12, 8.87)]),
                np.array([(-6.00, 1.73), (-6.60, 1.73), (-6.60, 2.89), (-7.20, 2.89),
                          (-7.20, 4.66), (-7.80, 4.66), (-7.80, 7.00), (-4.80, 7.00),
                          (-4.80, 4.66), (-5.40, 4.66), (-5.40, 2.89), (-6.00, 2.89)])
            ],
            'platform': {
                'outer': [
                    (-6.00, 1.73), (-6.60, 1.73), (-6.60, 2.89), (-7.20, 2.89),
                    (-7.20, 4.66), (-7.80, 4.66), (-7.80, 7.00), (-4.80, 7.00),
                    (-4.80, 4.66), (-5.40, 4.66), (-5.40, 2.89), (-6.00, 2.89)
                ],
                'inner': [
                    (-6.78, 3.31), (-6.78, 5.08), (-7.38, 5.08),
                    (-7.38, 6.58), (-5.22, 6.58), (-5.22, 5.08),
                    (-5.82, 5.08), (-5.82, 3.31)
                ]
            }
        },
        15: {
            1: {
                'center_point': np.array((-6.14, 5.05)),
                'corner_points': [
                    #main room boundary
                    np.array([(-8.12, 8.87), (-4.30, 8.87), (-4.30, 0.98), (-8.12, 0.98), (-8.12, 8.87)]),
                    #down wall wall 1
                    np.array([(-7.36, 2.60), (-7.36, 2.30), (-7.96, 2.30), (-7.96, 2.60)]),
                    #wall 2 side wall
                    np.array([
                        (-6.06, 3.09),
                        (-6.25, 3.09),
                        (-6.25, 3.91),
                        (-5.31, 3.91),
                        (-5.31, 6.26),
                        (-5.98, 6.26),
                        (-5.98, 6.44),
                        (-5.25, 6.44),
                        (-5.25, 3.72),
                        (-6.03, 3.72),
                        (-6.03, 3.09)
                    ]),
                    #wall 3 upper wall
                    np.array([(-4.30, 7.81), (-4.30, 7.60), (-6.90, 7.60), (-6.90, 7.81)]),
                    #wall 4 t shape
                    np.array([(-7.06, 5.95), (-7.25, 5.95), (-7.25, 5.16), (-8.02, 5.16),
                              (-8.02, 4.95), (-7.25, 4.95), (-7.25, 4.31), (-7.06, 4.31),
                              (-7.06, 4.95), (-6.04, 4.95), (-6.04, 5.16), (-7.06, 5.16),
                              (-7.06, 5.95)])
                ]
            },
            2: {
                'center_point': np.array((-5.14, 9.05)),
                'corner_points': [
                    #main room boundary
                    np.array([(-8.12, 8.87), (-4.30, 8.87), (-4.30, 0.98), (-8.12, 0.98), (-8.12, 8.87)]),
                    #down wall wall 1
                    np.array([(-8.10, 2.25), (-5.67, 2.25), (-5.67, 2.05), (-8.10, 2.05)]),
                    #wall 2 side wall
                    np.array([(-6.00, 6.45), (-5.22, 6.45) ,(-5.22, 3.74),(-6.04, 3.74) ,(-6.05, 3.09) ,(-6.25, 3.09) ,(-6.25, 3.93) ,(-5.28, 3.93) ,(-5.28, 6.26),(-6.00, 6.26)]),
                    #wall 3 upper wall
                    np.array([(-6.92, 7.81), (-4.31, 7.81), (-4.31, 7.60),(-6.92, 7.60)]),
                    #wall 4 t shape
                    np.array([(-7.26, 5.95), (-7.06, 5.95), (-7.06, 5.18), (-6.04, 5.18),
                              (-6.04, 4.96), (-7.06, 4.96), (-7.06, 4.31), (-7.26, 4.31),
                              (-7.26, 4.97), (-8.02, 4.97), (-8.02, 5.18), (-7.26, 5.18)])
                ]
            }
        }
    }
    if trial_num == 15:
        return coords.get(trial_num, {}).get(version, coords[trial_num][1])
    return coords.get(trial_num)

def main(segment_length=90, step_size=1, participant_sessions=None, filename_prefix="segment_data"):
    vr_base_dir = os.path.join(os.getcwd(), "data", "vr")
    ecg_base_dir= os.path.join(os.getcwd(), "data")
    base_dir = os.path.join(os.getcwd(), "data")
    trials = [7, 11, 15]

    segment_data_list = []

    for participant_info in participant_sessions:
        participant_id = participant_info[0]
        session_id = participant_info[1]
        version = participant_info[2] if len(participant_info) > 2 else 1

        ecg_dir = find_latest_ecg_directory(ecg_base_dir, participant_id, session_id)
        # corrected r peaks by sofie and sonja has session id as visit id. hence the mapping session 1 to visit 2 
        #and session 2 to visit 4
        if session_id == 1:
            session_str = "02"
        elif session_id == 2:
            session_str = "04"
        else:
            raise ValueError(f"Unexpected session_id: {session_id}. Expected 1 or 2.")
        participant_id_short = participant_id[3:]
        corrected_r_peaks_file_pattern = os.path.join(
            base_dir, "ecg", "electrophysiological_recordings_cleaned", "VR", f"{participant_id_short}_{session_str}_VR_r_peaks_edited_latest.npy"
        )
        if corrected_r_peaks_file_pattern is None:
            print(f"[WARNING] No ECG directory found for participant {participant_id_short} in session {session_str}.")
            continue

        for trial in trials:
            # movement_path = os.path.join(
            #     base_dir, 'vr', str(participant_id), 'virtual_reality',
            #     f'S{session_id:03d}', 'trackers_rotated',
            #     f'camera_movement_T{trial:03d}.csv'
            # )
            movement_path = os.path.join(base_dir, 'RAW1', str(participant_id), 
                                       f'S{session_id:03d}', 'trackers_rotated', 
                                       f'camera_movement_T{trial:03d}.csv')

            if not os.path.exists(movement_path):
                print(f"[WARNING] Movement file not found: {movement_path}")
                continue

            coords = get_trial_coords(trial, version)
            movement_data = pd.read_csv(movement_path)
            #print(participant_id, trial, movement_data.columns.tolist())
            if movement_data.empty:
                print(f"[WARNING] Empty movement file: {movement_path}")
                continue

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
                            timestamp_diff = (utc_df['utc_timestamp'] - utc_df['steady_timestamp']).astype(int).to_numpy()[0]
                            heartrate_timestamps = ((heartrate_data['steady_timestamp'] + timestamp_diff) / 1e6).astype(int).to_numpy()

                            timestamps = pd.to_datetime(movement_data['timestamp']).apply(lambda x: x.timestamp()).astype(int).to_numpy()
                            movement_min = np.min(timestamps)
                            movement_max = np.max(timestamps)

                            mask = (heartrate_timestamps >= movement_min) & (heartrate_timestamps <= movement_max)
                            heartrate_data_aligned = heartrate_data[mask]
                            heartrate_timestamps_aligned = heartrate_timestamps[mask]
                            signal_al = heartrate_data_aligned['ExG [1]-ch1']
                    except Exception as e:
                        print(f"[ERROR] ECG processing error for {participant_id}: {str(e)}")

            timestamps = pd.to_datetime(movement_data['timestamp']).apply(lambda x: x.timestamp()).astype(int).to_numpy()
            if len(timestamps) == 0:
                print(f"[WARNING] No valid timestamps in file: {movement_path}")
                continue

            start_time = np.floor(np.min(timestamps))
            if trial == 11:
                start_time += 30
            end_time = start_time + 90

            segment_starts = np.arange(start_time, end_time - segment_length + 1, step=step_size)
            segment_ends = segment_starts + segment_length

            for idx, (start, end) in enumerate(zip(segment_starts, segment_ends)):
                movement_segment = movement_data[(timestamps >= start) & (timestamps <= end)]

                rmssd = np.nan
                hr = np.nan

                if signal_al is not None and heartrate_timestamps_aligned is not None:
                    r_peaks_pattern = corrected_r_peaks_file_pattern
                    corrected_r_peaks_files = glob.glob(r_peaks_pattern)

                    if corrected_r_peaks_files:
                        corrected_r_peaks_file = corrected_r_peaks_files[0]
                        corrected_r_peaks = np.load(corrected_r_peaks_file)
                        start_sample = int((start - np.min(timestamps)) * 256)
                        end_sample = int((end - np.min(timestamps)) * 256)
                        segment_r_peak_indices = corrected_r_peaks[
                            (corrected_r_peaks >= start_sample) & (corrected_r_peaks <= end_sample)
                        ]

                        if len(segment_r_peak_indices) >= 2:
                            rri, _ = peaks_to_rri(peaks=segment_r_peak_indices, sampling_rate=256, interpolate=True, filter_outliers=True)
                            rmssd = calc_rmssd(rri)
                            hr = get_segment_heart_rate(rri)
                else:
                    print('signal_al or heart_rate_timestamps_aligned is None')

                safe_area_coverage = np.nan
                unsafe_area_coverage = np.nan
                safe_time_spent=np.nan
                unsafe_time_spent=np.nan
                outside_time_spent=np.nan

                if trial == 11:
                    try:
                        outer_platform = Polygon(coords['platform']['outer'])
                        inner_platform = Polygon(coords['platform']['inner'])
                        total_platform = Polygon(coords['corner_points'][0])

                        coverage = get_segment_mean_area_covered_platform(movement_segment, outer_platform, inner_platform, total_platform)
                        safe_area_coverage = coverage['safe_area_coverage']
                        unsafe_area_coverage = coverage['unsafe_area_coverage']

                        spent= calculate_time_in_areas_platform(movement_segment, outer_platform, inner_platform)
                        safe_time_spent = spent['time_in_safe_area']
                        unsafe_time_spent = spent['time_in_unsafe_area']
                        outside_time_spent = spent['time_in_outside_area']
                    except Exception as e:
                        print(f"[WARNING] Platform area coverage error: {e}")

                if len(movement_segment) > 0:
                    segment_data_list.append({
                        'participant_id': participant_id,
                        'session': session_id,
                        'trial': trial,
                        'version': version,
                        'segment': idx,
                        'center_dist': get_segment_mean_center_dist(movement_segment, coords['center_point']),
                        'edge_dist': get_segment_mean_edge_dist(movement_segment, coords['corner_points']),
                        'speed': get_segment_speed(movement_segment),
                        'acceleration': get_segment_acceleration(movement_segment),
                        'stops_count': get_segment_stops_count(movement_segment),
                        'stops_duration': get_segment_stops_duration(movement_segment),
                        'max_distance': get_max_distance(movement_segment),
                        'area_covered': get_segment_mean_area_covered(movement_segment, coords['corner_points']),
                        'rmssd': rmssd,
                        'hr': hr,
                        'safe_area_coverage': safe_area_coverage,
                        'unsafe_area_coverage': unsafe_area_coverage,
                        'safe_time_spent': safe_time_spent,
                        'unsafe_time_spent': unsafe_time_spent,
                        'outside_time_spent': outside_time_spent
                    })

    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    segment_data = pd.DataFrame(segment_data_list)
    file_path = os.path.join(results_dir, f'{filename_prefix}_{segment_length}s_{step_size}s.csv')
    segment_data.to_csv(file_path, index=False)
    print(f"Saved results to: {file_path}")

    return segment_data

if __name__ == "__main__":
    segment_data = main(segment_length=90, step_size=1)
