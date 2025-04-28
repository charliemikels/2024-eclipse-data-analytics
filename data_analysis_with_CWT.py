
import pandas as pd
import matplotlib as matplotlib
import matplotlib.pyplot as plt
import scipy as scipy
import seaborn as sns
import numpy as np
import math
import os
import scipy.signal as signal
import pywt
import matplotlib.dates



# Load and clean input data ---------------------------------------------------
df = pd.read_csv("eclipseData.csv", low_memory=False)
# PercentTotality has the string "Out of Path" mixed into some of it's data. (See Balloon == "Eclipse Pod v2 - PFW Eng")
# The mixed data causes Pandas to see the entire column as an object, and not as a numeric, messing up graph rendering and sorting.
df.loc[:, "PercentTotality"] = pd.to_numeric(df["PercentTotality"], errors="coerce").fillna(-1)

# Timestamp is a string. Force it into a datetime so that the graphs don't try to label every timestamp tick mark.
df['Timestamp'] = pd.to_datetime(df['Timestamp'])


# -- New / Calculated columns -------------------------------------------------

# Custom scale altitude from 0-1 to fit with eclipse progress
df["AltProgress"] = (df["AltImputed"] - df["AltImputed"].min()) / (df["AltImputed"].max() - df["AltImputed"].min())

# Wind Data ("WindSpeed (m/s)", "MeridionalWindSpeed", "ZonalWindSpeed")
# Using the Haversine formula, calculate the horizontal windspeed based on the balloon's lat and lon.
# Accounts for the Earth's radious at each lat/lon.
# Wind speed will appear constant at spots if using linearly interpolated data.
def calculate_wind_data(windless_df):
    earth_radius_norm=6371000
    wind_balloon_dfs = []
    for balloon_name, balloon_df in windless_df.groupby('Balloon'):
        balloon_df['earth_radius_at_balloon'] = \
            earth_radius_norm * (1-((np.sin(np.radians(balloon_df['AdjLatFilled']))**2)/298.257223563))

        balloon_df['change_in_AdjLatFilled'] = balloon_df['AdjLatFilled'] - balloon_df['AdjLatFilled'].shift(1)
        balloon_df['change_in_AdjLonFilled'] = balloon_df['AdjLonFilled'] - balloon_df['AdjLonFilled'].shift(1)

        # Haversine formula components
        a =   np.sin(np.radians(balloon_df['change_in_AdjLatFilled'] / 2)) ** 2 \
            + np.cos(np.radians(balloon_df['AdjLatFilled']))                    \
            * np.cos(np.radians(balloon_df['AdjLatFilled'].shift(1)))           \
            * np.sin(np.radians(balloon_df['change_in_AdjLonFilled']) / 2) ** 2
        movement_along_circumference = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        balloon_df['change_in_distance'] = balloon_df['earth_radius_at_balloon'] * movement_along_circumference

        change_in_time = (balloon_df['Timestamp'] - balloon_df['Timestamp'].shift(1)).dt.total_seconds().replace(0, np.nan)
        balloon_df['WindSpeed (m/s)'] = (balloon_df['change_in_distance'])/(change_in_time)

        balloon_df['MeridionalWindSpeed'] = (balloon_df['change_in_AdjLatFilled'] * 110947.2) / (change_in_time)
        balloon_df['ZonalWindSpeed'] = np.sqrt(np.maximum(0, balloon_df['WindSpeed (m/s)']**2 - balloon_df['MeridionalWindSpeed']**2))

        # Remove the change_in_ columns to keep the returned data clean.
        balloon_df = balloon_df.drop(['change_in_AdjLatFilled', 'change_in_AdjLonFilled', 'change_in_distance'], axis=1)

        wind_balloon_dfs.append(balloon_df)

    return pd.concat(wind_balloon_dfs, ignore_index=True)

df = calculate_wind_data(df)


def plot_cross_power_spectral_density(b, temp, wind_speed, sampling_rate):
    """
    Function: plot_cross_power_spectral_density()
    
    Description:
    Computes and visualizes the Cross Power Spectral Density (CPSD) between temperature and wind speed signals 
    for a given balloon flight. The function filters the frequency range to focus on values between 0.001 Hz 
    and 0.01 Hz, identifying the frequency with the highest power.
    
    Steps:
    1. Compute the CPSD using SciPy's 'signal.csd' function.
    2. Filter the computed CPSD to retain frequencies within the range [0.001 Hz, 0.01 Hz].
    3. Identify the frequency corresponding to the maximum power within this range.
    4. Print the frequency with the highest power.
    5. Generate a plot of the filtered CPSD data.
    
    Parameters:
    - 'b': String identifier for the balloon flight.
    - 'temp': NumPy array containing temperature signal data.
    - 'wind_speed': NumPy array containing wind speed signal data.
    - 'sampling_rate': Sampling rate in Hz (samples per second).
    
    
    Returns:
    - A plot displaying the CPSD magnitude over the specified frequency range.
    - Printed output indicating the frequency with maximum spectral power within the filtered range.
    """

    # Calculate CPSD using scipy.signal.csd
    freq, Pxy = signal.csd(temp, wind_speed, fs=sampling_rate, nperseg=1250)
    
    # Filter frequencies in the desired range
    valid_indices = (freq >= 0.001) & (freq <= 0.01)
    freq_filtered = freq[valid_indices]
    Pxy_filtered = Pxy[valid_indices]

    
    # Find the index of the maximum value in Pxy
    max_index = np.argmax(np.abs(Pxy_filtered))  # Use np.abs to ensure you get the magnitude of the complex values

    
    # Get the corresponding frequency
    max_frequency = freq_filtered[max_index]
    
    print(f'frequency with max power: {max_frequency}')
    
    # Plot CPSD
    plt.figure(figsize=(10, 6))
    plt.plot(freq_filtered, np.abs(Pxy_filtered), label='CPSD |temp & wind_speed|') 
    plt.xlabel('Frequency (Hz)')
    plt.xlim(.001,.01)
    plt.ylabel('CPSD Magnitude')
    plt.title(f'{b} (CPSD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def butterworth_filter(bal, data, time, cutoff_frequency=0.1, sampling_rate=1.0, order=2):
    """Applies a Butterworth low-pass filter to data.

    Args:
        bal: Title for the plot.
        data: NumPy array of data.
        time: NumPy array of time values.
        cutoff_frequency: Cutoff frequency (Hz).
        sampling_rate: Sampling rate (Hz).
        order: Filter order.

    Returns:
        Filtered data.
    """
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_frequency / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = signal.lfilter(b, a, data)

# =============================================================================
#     # Plotting filtered data vs original data
#     plt.figure(figsize=(10, 6))
#     plt.plot(time, data, label='Original Data')
#     plt.plot(time, filtered_data, label='Filtered Data', linestyle='--')
#     plt.xlabel('Time')
#     plt.ylabel('Data')
#     plt.legend()
#     plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
#     plt.gca().xaxis.set_major_locator(matplotlib.dates.MinuteLocator(byminute=range(0, 60, 10)))
#     plt.title(bal + ' Butterworth Low-Pass Filter')
#     plt.grid(True)
#     plt.show()
# =============================================================================

    return filtered_data



def interpolate_balloons_to_two_second_intervals(df, method='linear'):
    """
    Interpolates balloon data to two-second intervals using the specified method.

    Parameters:
    - df: DataFrame containing balloon data, with columns 'Timestamp' and 'Balloon'.
    - method: String specifying the interpolation method to use (default: 'linear').

    Returns:
    - df_interpolated: DataFrame with interpolated data at two-second intervals.
    """
    import pandas as pd

    interpolated_balloon_dfs = []

    for balloon_name, balloon_df in df.groupby('Balloon'):
        # Create timestamp range with 2-second intervals
        start_time = balloon_df['Timestamp'].min()
        end_time = balloon_df['Timestamp'].max()
        complete_timestamps = pd.date_range(start=start_time, end=end_time, freq='2s')

        # Create new DataFrame with complete timestamps
        populated_timestamps_df = pd.DataFrame({'Timestamp': complete_timestamps})
        populated_timestamps_df['Balloon'] = balloon_name

        # Combine original data and new timestamps
        merged_df = pd.merge(
            left=populated_timestamps_df,
            right=balloon_df,
            on=['Timestamp', 'Balloon'],
            how='outer'  # Keeps original data to allow interpolation
        )

        # Interpolate using the specified method
        merged_df = merged_df.infer_objects(copy=False)
        merged_df = merged_df.interpolate(method=method)

        # Purge original data that does not align with the complete timestamps
        merged_df = merged_df[merged_df['Timestamp'].isin(complete_timestamps)]

        # Save balloon's data
        interpolated_balloon_dfs.append(merged_df)

    # Combine all balloons back together
    df_interpolated = pd.concat(interpolated_balloon_dfs, ignore_index=True)
    return df_interpolated
# df['TempImputed'] = df['TempImputed']-df['TempImputed'].median()

df_interpolated = interpolate_balloons_to_two_second_intervals(df)


def discrete(time, b, signal):
    """
    Applies the Discrete Wavelet Transform (DWT) using the Daubechies-4 (db4) wavelet to analyze and reconstruct 
    a signal using approximation coefficients.

    Parameters:
    ----------
    time : array-like
        Time values corresponding to the signal samples.
    b : str
        Label for the data being processed.
    signal : array-like
        The input signal to be transformed.

    Process:
    --------
    1. Converts `signal` to a NumPy array for compatibility.
    2. Computes the DWT using the db4 wavelet, producing approximation (`cA`) and detail (`cD`) coefficients.
    3. Modifies the coefficient list, keeping only the approximation coefficients while nullifying detail coefficients.
    4. Reconstructs the signal using inverse wavelet transform (`pywt.waverec`), highlighting low-frequency components.
    5. Trims `reconstructed_signal` to match the length of `time`.
    6. Generates three plots:
       - Original signal over time.
       - Approximation coefficients.
       - Reconstructed signal emphasizing low-frequency trends.

    Returns:
    --------
    reconstructed_signal : array-like
        The reconstructed signal, emphasizing lower-frequency components.
   """
   
    # Perform the Discrete Wavelet Transform
    wavelet = 'db4'  # Daubechies wavelet
    signal = np.asarray(signal)
    coeffs = pywt.dwt(signal, wavelet)
    cA, cD = coeffs  # Approximation and detail coefficients
    coeffs = [cA, None]  # Replace None with zeroed-out detail coefficients

    # Reconstruct signal from approximation coefficients only
    reconstructed_signal = pywt.waverec([(cA)] + [np.zeros_like(cD)], wavelet='db4')
    reconstructed_signal = reconstructed_signal[:len(time)]
    # Plot the original signal, approximation, and detail
    plt.figure(figsize=(9, 8))
    plt.subplot(3, 1, 1)
    plt.plot(time, signal, label=b+'')
    plt.title(b+' Data')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(cA, label=b+' Approximation Coefficients')
    plt.title('Approximation Coefficients (Low Frequency)')
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(time, reconstructed_signal, label=b+' Reconstructed')
    plt.title('Reconstructed')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    return reconstructed_signal
    
def normalize(data):
    norm = []
    for i in data:
        norm.append((i-data.min()) / (data.max()-data.min()))
    return np.array(norm)


balloons = ['Eclipse Pod v2 - AMA Pre', 'Eclipse Pod v2 - Eastbrook', 'Eclipse Pod v2 - IMS During', 'Eclipse Pod v2 - IMS Pre', 'Eclipse Pod v2 - IN Academy', 'Eclipse Pod v2 - Linton-S', 'Eclipse Pod v2 - NSE Upland', 'Eclipse Pod v2 - PFW Edu', 'Eclipse Pod v2 - PFW Eng', 'Eclipse Pod v2 - Stockbridge', 'Eclipse Pod v2 - Taylor U']
def main():
    '''
    Function: main()
    
    Description:
    This function processes atmospheric data collected from high-altitude balloon flights, specifically those labeled 
    'Eclipse Pod v2 - IMS During' and 'Eclipse Pod v2 - IMS Pre'. It filters the data based on altitude, applies signal 
    processing techniques, and performs a Continuous Wavelet Transform (CWT) to visualize temperature variations.
    
    Steps:
    1. Group data by 'Balloon' type.
    2. Select flights labeled 'Eclipse Pod v2 - IMS During' or 'Eclipse Pod v2 - IMS Pre'.
    3. Identify the time range when altitude exceeds 36,000 feet.
    4. Extract relevant time-series data for temperature ('TempImputed') and wind speed ('WindSpeed (m/s)').
    5. Apply a Butterworth filter for smoothing.
    6. Normalize temperature and wind speed data.
    7. Trim the first five data points (stabilization).
    8. Compute and plot the cross-power spectral density of temperature and wind speed.
    9. Perform a Continuous Wavelet Transform (CWT) using a Complex Morlet ('cmor1.5-1.0') wavelet.
    10. Normalize and plot the scalogram to observe time-frequency variations in temperature data.
    
    '''
    for b in df_interpolated.groupby(['Balloon']):
        if b[0][0] == 'Eclipse Pod v2 - IMS During' or b[0][0] == 'Eclipse Pod v2 - IMS Pre':
            minima = b[1][b[1]['AltImputed']>=36000]['TimestampinSecs'].min()
            maxima = b[1][b[1]['AltImputed']>=36000]['TimestampinSecs'].max()
            df_eclipse = b[1][(b[1]['TimestampinSecs'] >= minima) & (b[1]['TimestampinSecs'] <= maxima)]
            time = np.array(df_eclipse['Timestamp'])
            data = np.array(df_eclipse['TempImputed'])
            data = butterworth_filter(b[0][0], data, time, cutoff_frequency=0.1, sampling_rate=0.5, order=2)
            y = np.array(df_eclipse['WindSpeed (m/s)'])
            y = butterworth_filter(b[0][0], y, time, cutoff_frequency=0.1, sampling_rate=0.5, order=2)
            data=normalize(data)
            y=normalize(y)
            data = data[5:]
            y = y[5:]
            time=time[5:]
    
            plot_cross_power_spectral_density(b[0][0], data, y, 0.5)
        
        
            # Define the wavelet and scales
            wavelet = 'cmor1.5-1.0'  # Complex Morlet wavelet
            scales = np.arange(100, 1000)
            
            
            # Perform the CWT
            coefficients, frequencies = pywt.cwt(data, scales, wavelet, sampling_period=2)
    
            # Normalize the coefficients
            coefficients = coefficients / np.max(np.abs(coefficients))
        
            # Plot the original signal and the CWT as a scalogram
            plt.figure(figsize=(10, 8))
        
            # Original Signal
            plt.subplot(2, 1, 1)
            plt.plot(time, data)
            plt.title(b[0][0]+ ' Temperature')
            plt.xlabel('Time')
            plt.ylabel('Temperture (Normalized)')
        
            # Continuous Wavelet Transform
            plt.subplot(2, 1, 2)
            plt.imshow(np.abs(coefficients), extent=[time[0], time[-1], scales[0], scales[-1]], 
                       aspect='auto', cmap='viridis')
            plt.title(b[0][0]+ ' Continuous Wavelet Transform (Scalogram) ')
            plt.xlabel('Time')
            plt.grid(False)
            plt.ylabel('Scale')
            plt.colorbar(label='Magnitude')
        
            plt.tight_layout()
            plt.show()
    return
main()

