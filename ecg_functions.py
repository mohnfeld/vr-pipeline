import neurokit2 as nk
import numpy as np
def peaks_to_rri(peaks=None, sampling_rate=256, interpolate=True, filter_outliers=True, method='quadratic'):

    peaks=np.array(peaks)
    rri = np.diff(peaks) / sampling_rate * 1000
  

    if filter_outliers:
        # Calculate the mean and standard deviation of RRIs
        rri_mean = np.mean(rri)
        rri_std = np.std(rri)
        threshold = 2.5
        outlier_mask = (rri >= rri_mean - threshold * rri_std) & (rri <= rri_mean + threshold * rri_std)
        rri = rri[outlier_mask]
        peaks = peaks[:-1][outlier_mask]

        # Filter out RRIs that are beyond the threshold
        rri = rri[(rri >= rri_mean - threshold * rri_std) & (rri <= rri_mean + threshold * rri_std)]
    if interpolate is False:
        return rri, sampling_rate
    else:
            # Minimum sampling rate for interpolation
        if sampling_rate < 10:
            sampling_rate = 10
            # Compute length of interpolated heart period signal at requested sampling rate.
            desired_length = int(np.rint(peaks[-1]))
            rri = nk.signal_interpolate(
                peaks[1:],  # Skip first peak since it has no corresponding element in heart_period
                rri, method=method,  # Interpolation method #monotone_cubic
                x_new=np.arange(1, desired_length)  # Start from 1 to match the length of rri
            )
        return rri, sampling_rate
    

def calc_rmssd(rri):
    rr_diff_ms = np.diff(rri)
    
    # Square each value in rr_diff_ms
    rr_diff_squared = rr_diff_ms ** 2
    
    # Calculate the average of rr_diff_squared
    rr_diff_mean = np.mean(rr_diff_squared)
    
    # Calculate the square root of rr_diff_mean to get RMSSD
    rmssd = np.sqrt(rr_diff_mean)
    
    return rmssd

def get_segment_heart_rate(rri):
    rri= rri /1000
    bpm = 60 / np.mean(rri)
    return bpm