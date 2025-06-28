#from biosppy.signals.tools import filter_signal
import pandas as pd
#import biosppy.signals.ecg as ecg
import numpy as np
from shapely.geometry import Point, LineString
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from scipy.spatial import ConvexHull
#from calculations import calculate_accelerations_t7_t15, calculate_speeds_t7_t15


def dist(a, b):
    return np.sqrt(np.square(a-b).sum(axis=1))

def minimum_distance(v, w, p):
  l2 = (dist(v, w) ** 2)[0]
  if l2 == 0.0:
    return dist(p, v)
  t = np.clip(np.dot(p - v, np.transpose(w - v, [1,0])) / l2, 0, 1)
  projection = v + t * (w - v)
  return dist(p, projection)

def mean_center_distance(points, center_point):
    dists = dist(points, center_point)
    return np.mean(dists)

def mean_edge_distance(points, corner_points):
    if isinstance(corner_points, list):
        # Handle multiple wall segments
        all_distances = []
        for wall_segment in corner_points:
            dists = np.zeros((wall_segment.shape[0], len(points)))
            rotated_corner_points = np.roll(wall_segment, 1, axis=0)
            point_pairs = [(wall_segment[i], rotated_corner_points[i]) 
                          for i in range(len(wall_segment))]
            
            for idx, (p1, p2) in enumerate(point_pairs):
                dists[idx,:] = minimum_distance(np.expand_dims(p1,0),
                                             np.expand_dims(p2,0),
                                             points)
            # Get minimum distance to this wall segment
            all_distances.append(np.min(dists, axis=0))
        
        # Return minimum distance to any wall
        min_dist=np.min(np.array(all_distances), axis=0)
        return np.mean(min_dist)
    else:
        # Original behavior for single wall segment
        dists = np.zeros((corner_points.shape[0], len(points)))
        rotated_corner_points = np.roll(corner_points, 1)
        point_pairs = [(corner_points[i], rotated_corner_points[i]) 
                      for i in range(len(corner_points))]
        
        for idx, (p1,p2) in enumerate(point_pairs):
            dists[idx,:] = minimum_distance(np.expand_dims(p1,0),
                                          np.expand_dims(p2,0),
                                          points)
        return np.mean(np.min(dists, axis=0))

#center_point = np.array((-6.33, 5.14))
#corner_points = np.array([(-4,2),(-9,2),(-9,9),(-4,9)])
#corner_points=np.array([[-8.12, 8.87], [-4.30, 8.87], [-4.30, 0.98], [-8.12, 0.98], [-8.12, 8.87]]) # empty room coords
#print(mean_center_distance(points, center_point))
#print(mean_edge_distance(points, corner_points))

def get_segment(data, start_time, end_time, timestamps):
    segment_points = data[np.logical_and(timestamps >= start_time, 
                                         timestamps < end_time)]
    return segment_points
    
def get_segment_mean_center_dist(segment, center_point):
    # transform points into np array of dimension [N, 2]
    points = np.transpose(np.stack((segment['pos_x'], segment['pos_z'])), axes=[1,0])
    
    center_dist = mean_center_distance(points, center_point)
    return center_dist

def get_segment_mean_edge_dist(segment, corner_points):
    points = np.transpose(np.stack((segment['pos_x'], segment['pos_z'])), axes=[1,0])
    
    if isinstance(corner_points, list):
        # If corner_points is a list of arrays (multiple objects)
        distances = []
        for object_points in corner_points:
            dist = mean_edge_distance(points, object_points)
            distances.append(dist)
        return np.min(distances)  # Return the minimum distance to any object
    else:
        # If corner_points is a single array
        return mean_edge_distance(points, corner_points)

def get_segment_mean_area_covered(segment, corner_points):
    if isinstance(corner_points, list):
        # First element is the main room
        main_room = Polygon(corner_points[0])
        # Rest are walls/obstacles to subtract
        walls = [Polygon(wall) for wall in corner_points[1:]]
        
        # Subtract walls from main room to get navigable area
        navigable_area = main_room
        for wall in walls:
            navigable_area = navigable_area.difference(wall)
        
        # Create trajectory
        trajectory_line = LineString(zip(segment['pos_x'], segment['pos_z']))
        trajectory_area = trajectory_line.buffer(0.1)
        
        # Calculate intersection with navigable area
        total_intersection = trajectory_area.intersection(navigable_area)
        total_area = navigable_area.area
        
        percentage_covered = (total_intersection.area / total_area) * 100
        return percentage_covered
    else:
        # Original behavior for simple rooms
        room = Polygon(corner_points)
        trajectory_line = LineString(zip(segment['pos_x'], segment['pos_z']))
        trajectory_area = trajectory_line.buffer(0.1)
        trajectory_area = trajectory_area.area
        #total_intersection = trajectory_area.intersection(room)
        total_area = room.area
        #percentage_covered = (total_intersection.area / total_area) * 100
        percentage_covered = (trajectory_area / total_area) * 100
        return percentage_covered
    
def get_segment_mean_area_covered_platform(segment, outer_platform, inner_platform, total_platform):
    """Calculate percentage of each platform zone covered by the trajectory"""
    # Create trajectory buffer
    trajectory_line = LineString(zip(segment['pos_x'], segment['pos_z']))
    trajectory_area = trajectory_line.buffer(0.1)
    
    # Calculate areas for different zones
    outside_zone=total_platform.difference(outer_platform)
    safe_zone = inner_platform
    unsafe_zone = outer_platform.difference(inner_platform)
    
    # Calculate intersections
    safe_intersection = trajectory_area.intersection(safe_zone)
    unsafe_intersection = trajectory_area.intersection(unsafe_zone)
    outside_intersection= trajectory_area.intersection(outside_zone)
    # Calculate percentages

    safe_percentage = (safe_intersection.area / safe_zone.area) * 100
    unsafe_percentage = ((unsafe_intersection.area + outside_intersection.area) / unsafe_zone.area) * 100
    
    return {
        'safe_area_coverage': safe_percentage,
        'unsafe_area_coverage': unsafe_percentage
    }

def check_position_safety(x, z, outer_platform, inner_platform):
    """
    Determine if a position is in safe, unsafe, or outside area
    
    Returns:
    -1: outside platform
    0: unsafe area (between outer and inner)
    1: safe area (inside inner platform)
    """
    point = Point(x, z)
    
    if not point.within(outer_platform):
        return -1  # Outside platform
    elif point.within(inner_platform):
        return 1   # Safe area
    else:
        return 0   # Unsafe area

def calculate_time_in_areas_platform(segment, outer_platform, inner_platform):
    """
    Calculate time spent in safe, unsafe, and outside areas.
    Assumes segment contains 'pos_x', 'pos_z', and 'time' columns.
    """
    time_in_safe_area = 0
    time_in_unsafe_area = 0
    time_in_outside_area = 0
    
    times = segment['time'].values
    
    for i in range(1, len(segment)):
        time_diff = times[i] - times[i-1]
        safety = check_position_safety(
            segment['pos_x'].iloc[i], 
            segment['pos_z'].iloc[i], 
            outer_platform, 
            inner_platform
        )
        
        if safety == 1:  # Safe area (inner platform)
            time_in_safe_area += time_diff
        elif safety == 0:  # Unsafe area (between platforms)
            time_in_unsafe_area += time_diff
        else:  # Outside platforms
            time_in_outside_area += time_diff
            
    return {
        'time_in_safe_area': time_in_safe_area,
        'time_in_unsafe_area': time_in_unsafe_area,
        'time_in_outside_area': time_in_outside_area
    }
'''
def get_max_distance(segment):
    # Stack pos_x and pos_z into points array
    points = np.transpose(np.stack((segment['pos_x'], segment['pos_z'])), axes=[1,0])
    
    # Calculate maximum distance between any two points
    max_dist = 0
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            dist = np.sqrt(np.sum((points[i] - points[j]) ** 2))
            max_dist = max(max_dist, dist)
    return max_dist
'''
'''
def get_max_distance(segment):
    # Stack pos_x and pos_z into points array
    points = np.transpose(np.stack((segment['pos_x'], segment['pos_z'])), axes=[1,0])
    
    # Find min and max coordinates
    min_coords = np.min(points, axis=0)  # [min_x, min_z]
    max_coords = np.max(points, axis=0)  # [max_x, max_z]
    
    # Calculate maximum possible distance
    max_dist = np.sqrt(np.sum((max_coords - min_coords) ** 2))
    
    return max_dist
'''

def get_max_distance(segment):
    # Stack pos_x and pos_z into points array
    #points = np.vstack((segment['pos_x'], segment['pos_z'])).T
    points = np.transpose(np.stack((segment['pos_x'], segment['pos_z'])), axes=[1,0])
    
    # Compute the convex hull
    hull = ConvexHull(points)
    
    # Get the indices of the points on the convex hull
    hull_indices = hull.vertices
    
    # Extract the points on the convex hull
    hull_points = points[hull_indices]
    
    # Initialize variables for maximum distance and furthest pair
    max_distance = 0
    # Iterate over pairs of points on the convex hull
    for i in range(len(hull_points)):
        for j in range(i+1, len(hull_points)):
            dist = np.sqrt(np.sum((hull_points[i] - hull_points[j])**2))
            if dist > max_distance:
                max_distance = dist
    return max_distance

def get_segment_stops_duration(movement_segment, stop_threshold=0.10, stop_duration_threshold=0.5):
    # Calculate speeds within the segment
    diff_df = movement_segment.drop('timestamp', axis=1).diff(axis=0)
    #diff_df["dist"] = np.sqrt(diff_df["pos_x"].pow(2) + diff_df["pos_y"].pow(2) + diff_df["pos_z"].pow(2))
    diff_df["dist"] = np.sqrt(diff_df["pos_x"].pow(2) + diff_df["pos_z"].pow(2))
    speeds = diff_df['dist']/diff_df['time']
    
    # Get time values
    time = movement_segment['time'].to_numpy()
    
    stops = []
    current_stop_start = None

    for i, speed in enumerate(speeds):
        if speed < stop_threshold:
            if current_stop_start is None:
                current_stop_start = time[i]
        else:
            if current_stop_start is not None:
                stop_duration = time[i] - current_stop_start
                if stop_duration >= stop_duration_threshold:
                    stops.append(stop_duration)
                current_stop_start = None

    # Check for ongoing stop at the end
    if current_stop_start is not None:
        stop_duration = time[-1] - current_stop_start
        if stop_duration >= stop_duration_threshold:
            stops.append(stop_duration)
    
    # Return total duration of stops in the segment
    return sum(stops) if stops else 0


def get_segment_stops_count(movement_segment, stop_threshold=0.10, stop_duration_threshold=0.5):
    """Calculate number of stops within a segment of trajectory."""
    # Calculate speeds within the segment
    diff_df = movement_segment.drop('timestamp', axis=1).diff(axis=0)
   # diff_df["dist"] = np.sqrt(diff_df["pos_x"].pow(2) + diff_df["pos_y"].pow(2) + diff_df["pos_z"].pow(2))
    diff_df["dist"] = np.sqrt(diff_df["pos_x"].pow(2) + diff_df["pos_z"].pow(2))
    speeds = diff_df['dist']/diff_df['time']
    
    # Get time values
    time = movement_segment['time'].to_numpy()
    
    stops_count = 0
    current_stop_start = None

    for i, speed in enumerate(speeds):
        if speed < stop_threshold:
            if current_stop_start is None:
                current_stop_start = time[i]
        else:
            if current_stop_start is not None:
                stop_duration = time[i] - current_stop_start
                if stop_duration >= stop_duration_threshold:
                    stops_count += 1  
                current_stop_start = None

    # Check for ongoing stop at the end
    if current_stop_start is not None:
        stop_duration = time[-1] - current_stop_start
        if stop_duration >= stop_duration_threshold:
            stops_count += 1
    return stops_count  
'''
def get_segment_heart_rate(segment, sampling_rate=256):
    r_peaks = find_r_peaks(segment, sampling_rate)
    rr_intervals = np.diff(r_peaks) / sampling_rate
    bpm = 60 / np.mean(rr_intervals)
    return bpm

def preprocess_ecg(signal, sampling_rate):
    #denoised_signal = cwt_denoise(signal)
    # Preprocess the ECG signal
    # Bandpass filter the ECG signal (0.5-40 Hz)
    filtered_signal = filter_signal(signal,
    #                                denoised_signal,
                                    ftype='FIR',
                                    band='bandpass',
                                    order=int(0.3 * sampling_rate),
                                    frequency=[0.1, 40],
                                    sampling_rate=sampling_rate) # check if they are the same and hf is getting filtered
    filtered_signal = filtered_signal[0]
    #filtered_signal = filtered_signal.reshape(-1)
    return filtered_signal

def find_r_peaks(signal, sampling_rate):
    filtered_signal = preprocess_ecg(signal, sampling_rate)
    # Find R peaks using the Pan-Tompkins algorithm
    r_peaks = ecg.engzee_segmenter(filtered_signal, sampling_rate=sampling_rate)[0]
    return r_peaks

def filter_rr_intervals(rr_intervals):
    # Remove RR intervals that are physiologically impossible
    # For example, intervals < 300ms or > 2000ms
    return rr_intervals[(rr_intervals >= 0.3) & (rr_intervals <= 2.0)]
'''
'''
def calculate_rmssd(r_peaks, sampling_rate):
    # Calculate the RR intervals
    rr_intervals = np.diff(r_peaks) / sampling_rate
    
    # Calculate the successive differences of RR intervals
    rr_diff = np.diff(rr_intervals)
    
    # Calculate RMSSD
    rmssd = np.sqrt(np.mean(rr_diff ** 2))
    return rmssd
'''
'''
def calculate_rmssd(r_peaks, sampling_rate):
    # Calculate the RR intervals in milliseconds
    rr_intervals_ms = np.diff(r_peaks) / sampling_rate * 1000
    
    # Calculate the successive differences of RR intervals
    rr_diff_ms = np.diff(rr_intervals_ms)
    
    # Square each value in rr_diff_ms
    rr_diff_squared = rr_diff_ms ** 2
    
    # Calculate the average of rr_diff_squared
    rr_diff_mean = np.mean(rr_diff_squared)
    
    # Calculate the square root of rr_diff_mean to get RMSSD
    rmssd = np.sqrt(rr_diff_mean)
    
    return rmssd

def get_segment_rmssd(signal_segment, sampling_rate=256):
    r_peaks = find_r_peaks(signal_segment, sampling_rate)
    rmssd = calculate_rmssd(r_peaks, sampling_rate)
    return rmssd
    '''

def get_segment_speed(movement_data):    
    diff_df = movement_data.drop('timestamp', axis=1).diff(axis=0)
    #diff_df["dist"] = np.sqrt(diff_df["pos_x"].pow(2) + diff_df["pos_y"].pow(2) + diff_df["pos_z"].pow(2))
    diff_df["dist"] = np.sqrt(diff_df["pos_x"].pow(2) + diff_df["pos_z"].pow(2))
    speed = diff_df['dist']/diff_df['time']
    return np.mean(speed)


def get_segment_acceleration(movement_data):
    diff_df = movement_data.drop('timestamp', axis=1).diff(axis=0)
    #diff_df["dist"] = np.sqrt(diff_df["pos_x"].pow(2) + diff_df["pos_y"].pow(2) + diff_df["pos_z"].pow(2))
    diff_df["dist"] = np.sqrt(diff_df["pos_x"].pow(2) + diff_df["pos_z"].pow(2))
    speeds = diff_df['dist']/diff_df['time']

    acc = speeds.diff() / diff_df['time']
    return np.mean(acc)


# if __name__ == "__main__":
#     movement_data = pd.read_csv(movement_path)
#     heartrate_data = pd.read_csv(heartbeat_path)
#     utc_df = pd.read_csv(utc_path)
#     timestamp_diff = (utc_df['utc_timestamp']-utc_df['steady_timestamp']).astype(int).to_numpy()[0]

#     heartrate_timestamps = ((heartrate_data['steady_timestamp'] + timestamp_diff)/1e6).astype(int).to_numpy()

#     ecg_signal = heartrate_data['ExG [1]-ch1']
#     # signal = preprocess_ecg(ecg_signal, 256)
#     # r_peaks = find_r_peaks(signal, 250)
#     # rmssd = calculate_rmssd(r_peaks, 250)

#     segment_length=10
#     step_size=1

#     # heart rate data 
#     #heart_rate_data

#     timestamps = pd.to_datetime(movement_data['timestamp']).apply(lambda x: x.timestamp()).astype(int).to_numpy()

#     movement_min = np.min(timestamps)  # Start of movement data
#     movement_max = np.max(timestamps)  # End of movement data

#     # Step 2: Align heart rate data to movement data range

#     heartrate_data_aligned = heartrate_data[
#         (heartrate_timestamps >= movement_min) & (heartrate_timestamps <= movement_max)
#     ]
#     heartrate_timestamps_aligned = heartrate_timestamps[
#         (heartrate_timestamps >= movement_min) & (heartrate_timestamps <= movement_max)
#     ]

#     ecg_signal_al = heartrate_data_aligned['ExG [1]-ch1']
#     # signal_al = preprocess_ecg(ecg_signal_al, 250)
#     # r_peaks_al = find_r_peaks(signal_al, 250)
#     # rmssd_al = calculate_rmssd(r_peaks_al, 250)

#     start_time = np.floor(np.min(timestamps))

#     end_time = start_time + 90 #for elevated platofrm it should be bigger than that
#     segment_starts = np.arange(start_time, end_time-segment_length+1, step=step_size)
#     segment_ends = segment_starts + segment_length

#     segment_mean_center_dist = np.zeros(len(segment_starts))
#     segment_hr = np.zeros(len(segment_starts))
#     segment_mean_speed = np.zeros(len(segment_starts))
#     segment_rmssd = np.zeros(len(segment_starts))
#     segment_mean_edge_dist = np.zeros(len(segment_starts))
#     segment_mean_acc= np.zeros(len(segment_starts))
#     segment_mean_area_covered=np.zeros(len(segment_starts))
#     segment_stops_duration = np.zeros(len(segment_starts))
#     segment_stops_count= np.zeros(len(segment_starts))
#     segment_max_distance = np.zeros(len(segment_starts))

#     for idx, (start, end) in enumerate(zip(segment_starts, segment_ends)):
#         movement_segment = get_segment(movement_data, start, end, timestamps)
#         center_dist = get_segment_mean_center_dist(movement_segment, center_point)    
#         segment_mean_center_dist[idx] = center_dist
        
#         edge_dist = get_segment_mean_edge_dist(movement_segment, corner_points)
#         segment_mean_edge_dist[idx] = edge_dist
        
#         speed = get_segment_speed(movement_segment)
#         segment_mean_speed[idx] = speed

#         acc=get_segment_acceleration(movement_segment)
#         segment_mean_acc[idx]= acc
        
#         corner_points=np.array([[-8.12, 8.87], [-4.30, 8.87], [-4.30, 0.98], [-8.12, 0.98]]) #empty room coords
#         area_covered = get_segment_mean_area_covered(movement_segment, corner_points)
#         segment_mean_area_covered[idx] = area_covered

#         #hr_segment = get_segment(signal, start, end, heartrate_timestamps)
#         # hr_segment = get_segment(signal_al, start, end, timestamps=heartrate_timestamps_aligned)
#         # hr=get_segment_heart_rate(hr_segment)
#         # segment_hr[idx] = hr

#         stops_duration = get_segment_stops_duration(movement_segment, stop_threshold=0.10, stop_duration_threshold=2.0)
#         segment_stops_duration[idx] = stops_duration

#         stops_count= get_segment_stops_count(movement_segment, stop_threshold=0.10, stop_duration_threshold=2.0)
#         segment_stops_count[idx] = stops_count

#         distance_max = get_max_distance(movement_segment)
#         segment_max_distance[idx] = distance_max
        
#         #rmssd = get_segment_rmssd(segment_hr)
#         #segment_rmssd[idx] = rmssd 
    
