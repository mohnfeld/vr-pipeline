### original functions from participants_functions for some reason it stopped working

from segment_functions import *
from ecg_functions import *
import os
import pandas as pd
import numpy as np
import glob

def find_latest_ecg_directory(base_dir, participant_id):
    ecg_path_pattern = os.path.join(base_dir, str(participant_id), 'virtual_reality', 'BBT-BIO-AAB*')
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
            ]
        },
        15: {
            1: {
                'center_point': np.array((-6.14, 5.05)),
                'corner_points': [
                    np.array([(-8.12, 8.87), (-4.30, 8.87), (-4.30, 0.98), (-8.12, 0.98), (-8.12, 8.87)]),
                    np.array([(-7.36, 2.60), (-7.36, 2.30), (-7.96, 2.30), (-7.96, 2.60)]),
                    np.array([(-6.72, 4.90), (-6.72, 4.31), (-7.02, 4.31), (-7.02, 4.90)]),
                    np.array([(-4.30, 7.81), (-4.30, 7.60), (-6.90, 7.60), (-6.90, 7.81)]),
                    np.array([(-7.06, 5.95), (-7.25, 5.95), (-7.25, 5.16), (-8.02, 5.16),
                              (-8.02, 4.95), (-7.25, 4.95), (-7.25, 4.31), (-7.06, 4.31),
                              (-7.06, 4.95), (-6.04, 4.95), (-6.04, 5.16), (-7.06, 5.16),
                              (-7.06, 5.95)])
                ]
            },
            2: {
                'center_point': np.array((-5.14, 9.05)),
                'corner_points': [
                    np.array([(-8.12, 8.87), (-4.30, 8.87), (-4.30, 0.98), (-8.12, 0.98), (-8.12, 8.87)]),
                    np.array([(-4.52, 2.60), (-4.52, 2.30), (-3.92, 2.30), (-3.92, 2.60)]),
                    np.array([(-5.36, 4.90), (-5.36, 4.31), (-5.06, 4.31), (-5.06, 4.90)]),
                    np.array([(-4.30, 7.81), (-4.30, 7.60), (-5.38, 7.60), (-5.38, 7.81)]),
                    np.array([(-5.22, 5.95), (-5.03, 5.95), (-5.03, 5.16), (-4.26, 5.16),
                              (-4.26, 4.95), (-5.03, 4.95), (-5.03, 4.31), (-5.22, 4.31),
                              (-5.22, 4.95), (-6.24, 4.95), (-6.24, 5.16), (-5.22, 5.16),
                              (-5.22, 5.95)])
                ]
            }
        }
    }
    if trial_num == 15:
        return coords.get(trial_num, {}).get(version, coords[trial_num][1])
    return coords.get(trial_num)

def main(segment_length=90, step_size=1, participant_sessions=None, filename_prefix="segment_data"):
    vr_base_dir = os.path.join(os.getcwd(), "data", "vr")
    base_dir = os.path.join(os.getcwd(), "data")
    trials = [7, 11, 15]

    segment_data_list = []

    for participant_info in participant_sessions:
        participant_id = participant_info[0]
        session_id = participant_info[1]
        version = participant_info[2] if len(participant_info) > 2 else 1

        ecg_dir = find_latest_ecg_directory(vr_base_dir, participant_id)
        corrected_r_peaks_file_pattern = os.path.join(
            base_dir, "ecg", f"{{participant_id}}_{session_id}_{version}_r_peaks_edited_*.npy"
        )

        for trial in trials:
            movement_path = os.path.join(
                base_dir, 'vr', str(participant_id), 'virtual_reality',
                f'S{session_id:03d}', 'trackers_rotated',
                f'camera_movement_T{trial:03d}.csv'
            )

            if not os.path.exists(movement_path):
                print(f"[WARNING] Movement file not found: {movement_path}")
                continue

            coords = get_trial_coords(trial, version)
            movement_data = pd.read_csv(movement_path)
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
                    r_peaks_pattern = corrected_r_peaks_file_pattern.format(participant_id=participant_id)
                    print(f"[DEBUG] Searching R-peaks file with pattern: {r_peaks_pattern}")
                    corrected_r_peaks_files = glob.glob(r_peaks_pattern)

                    if corrected_r_peaks_files:
                        corrected_r_peaks_file = corrected_r_peaks_files[0]
                        print(f"[DEBUG] Found R-peaks file: {corrected_r_peaks_file}")
                        corrected_r_peaks = np.load(corrected_r_peaks_file)
                        start_sample = int((start - movement_min) * 256)
                        end_sample = int((end - movement_min) * 256)
                        segment_r_peak_indices = corrected_r_peaks[
                            (corrected_r_peaks >= start_sample) & (corrected_r_peaks <= end_sample)
                        ]

                        print(f"[DEBUG] Segment {idx} has {len(segment_r_peak_indices)} R-peaks")

                        if len(segment_r_peak_indices) >= 2:
                            rri, _ = peaks_to_rri(peaks=segment_r_peak_indices, sampling_rate=256, interpolate=True, filter_outliers=True)
                            rmssd = calc_rmssd(rri)
                            hr = get_segment_heart_rate(rri)
                        else:
                            print(f"[DEBUG] No valid R-peaks for segment {idx} (participant {participant_id}, trial {trial})")
                    else:
                        print(f"[DEBUG] No R-peaks file found for: {r_peaks_pattern}")

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
                        'hr': hr
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
