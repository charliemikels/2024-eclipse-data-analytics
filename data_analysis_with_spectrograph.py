import pandas as pd
import matplotlib as matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

# Config
matplotlib.use('agg')


# Load and clean input data ---------------------------------------------------
df = pd.read_csv("eclipseData.csv", low_memory=False)

# PercentTotality has the string "Out of Path" mixed into some of it's data. (See Balloon == "Eclipse Pod v2 - PFW Eng")
# The mixed data causes Pandas to see the entire column as an object, and not as a numeric, messing up graph rendering and sorting.
df["PercentTotality"] = pd.to_numeric(df["PercentTotality"], errors="coerce")
df["PercentTotality"] = df["PercentTotality"].fillna(-1)    # daisy chaining threw a diagnostic error

# Timestamp is a string. Force it into a datetime so that the graphs don't try to label every timestamp tick mark.
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Trim Data to just our altitude range
min_alt_limit = 36_000
if min_alt_limit:
    df = df[df["AltImputed"] >= min_alt_limit]


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

df_with_wind = calculate_wind_data(df)


# Linear interpolation --------------------------------------------------------

# Some analysis methods require consistent time steps, and our data is sometimes missing rows.
# Interpolation forces our data into consistent time steps.
#
# Group by balloon, enforce timestamps to actualy arrive exactly every 2 seconds
# Fill in data for created timestamps using linear interpolation.
# Remove original data that does not line up on the 2 second intervals
# Merge interpolated data back together
#
# Some data, like NA values and packet sequence numbers, are not interpolated.
def interpolate_balloons_to_two_second_intervals(df, freq='2s'):

    # Ensure 'Timestamp' is of type datetime. Used for pd.date_range later
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Ensure timestamps are sorted for each balloon
    df = df.sort_values(['Balloon', 'Timestamp'])

    interpolated_balloon_dfs = []
    for balloon_name, balloon_df in df.groupby('Balloon'):
        # Create timestamp range with 2 second intervals
        start_time = balloon_df['Timestamp'].min()
        end_time = balloon_df['Timestamp'].max()
        complete_timestamps = pd.date_range(start=start_time, end=end_time, freq=freq)

        # Create new dataframe with complete timestamps
        populated_timestamps_df = pd.DataFrame({'Timestamp': complete_timestamps})
        populated_timestamps_df['Balloon'] = balloon_name

        # Combine original data and new timestamps. Creates collumns and nulls to match original data.
        merged_df = pd.merge(
            left=populated_timestamps_df,
            right=balloon_df,
            on=['Timestamp', 'Balloon'],
            how='outer' # "Outer" so that the original data is still present.
                        # sometimes the 2 second timestamps don't line up, so we need the original data for interpolation.
        )

        # Identify numeric columns (exclude datetime (already accounted for) and string columns)
        numeric_columns = merged_df.select_dtypes(include=['float64', 'int64']).columns

        # Interpolate original data into new timestamps.
        merged_df[numeric_columns] = merged_df[numeric_columns].interpolate(method='linear')

        # Purge original data that does not line up with complete_timestamps
        # This does mean we're sacrificing original data, but said data is accounted for during interpolation, and
        # this process ensures all timestamps (per balloon) are in 2 second intervals.
        merged_df = merged_df[merged_df['Timestamp'].isin(complete_timestamps.to_series())]

        # Save balloon's data.
        interpolated_balloon_dfs.append(merged_df)

    # Combine all balloons back together
    df_interpolated = pd.concat(interpolated_balloon_dfs, ignore_index=True)

    # df_interpolated.to_csv('interpolated.csv', index=False)
    return df_interpolated

df_interpolated = interpolate_balloons_to_two_second_intervals(df_with_wind, freq='2s')

# Adds flight and eclips highlights to a chart takes in a filtered df and a chart's AX
def add_eclipse_and_flight_progress_highlights(df, ax):
    eclipse_in_progress = df[df["PercentTotality"] > 0]
    if len(eclipse_in_progress) > 0:
        eclipse_start = eclipse_in_progress["Timestamp"].iloc[0]
        eclipse_end = eclipse_in_progress["Timestamp"].iloc[-1]
        ax.axvspan(xmin=eclipse_start, xmax=eclipse_end, alpha=0.05, color="grey")

    totality_in_progress = df[df["PercentTotality"] >= 1]
    if len(totality_in_progress) > 0:
        totality_start = totality_in_progress["Timestamp"].iloc[0]
        totality_end = totality_in_progress["Timestamp"].iloc[-1]
        ax.axvspan(xmin=totality_start, xmax=totality_end, alpha=0.1, color="black")

    ascent_in_progress = df[(df["Ascent"] == 1) & (df["AltProgress"] > 0.005)]
    if len(ascent_in_progress) > 0:
        ascent_start = ascent_in_progress["Timestamp"].iloc[0]
        ascent_end = ascent_in_progress["Timestamp"].iloc[-1]
        ax.axvspan(xmin=ascent_start, xmax=ascent_end, ymax=0.02, alpha=0.1, color="green")

    descent_in_progress = df[(df["Ascent"] == 0) & (df["AltProgress"] > 0.005)]
    if len(descent_in_progress) > 0:
        descent_start = descent_in_progress["Timestamp"].iloc[0]
        descent_end = descent_in_progress["Timestamp"].iloc[-1]
        ax.axvspan(xmin=descent_start, xmax=descent_end, ymax=0.02, alpha=0.1, color="red")

    if len(ascent_in_progress) > 0 and len(descent_in_progress) > 0:
        ascent_end = ascent_in_progress["Timestamp"].iloc[-1]
        descent_start = descent_in_progress["Timestamp"].iloc[0]
        ax.axvspan(xmin=ascent_end, xmax=descent_start, ymax=0.02, alpha=0.1, color="blue")

    return ax

# Spectrograph
# Generates and saves a spectrograph for the data using Enhanced Autocoralation
def generate_spectrograph_and_scatterplot(df, column_name, title_ending, filename):
    data = df[column_name].values

    # Calculate sampling. Assumes data is evenly spaced.
    seconds_per_sample = np.mean(np.diff(df['TimestampinSecs'].values)) # should be `2` after normal interpolation.
    segment_length = min(2048, len(data) * 2 // 3 )  # Smaller = temporal resolution. Higher = frequency resolution.
        # 1024 - Audacity default
        # A lot of our balloons within the right altatude have too few samples for the high window sizes, so `len(data)//2` is a fallback.

    segment_overlap_pct = 99 / 100
    segment_overlap_size = int(segment_length * segment_overlap_pct)   # huge overlap for more temporal resolution
    next_segment_step_size = segment_length - segment_overlap_size

    window_functions = {
        "rectangular":  np.ones(segment_length),
        "hanning":      np.hanning(segment_length),
        "hamming":      np.hamming(segment_length),
        "blackman":     np.blackman(segment_length),
        "kaiser_14":    np.kaiser(segment_length, beta=14)
    }
    window = window_functions["hamming"]

    # Get segments
    # A segment is a subset of our data with length `segment_length`.
    # each segment in `segments` is ofset from the one before by 1 sample.
    segments = []
    for i in range(0, len(data)-segment_length, next_segment_step_size):
        segment = (data[i:i+segment_length]) * window
        if len(segment) == segment_length:
            segments.append(segment * window)

    # Generate spectrogram using Enhanced Autocorrelation
    # 1. correlate slides a segment over itself to find periodic patterns within itself.
    # 2. FFT detects the strengths of peaks found by correlate.
    spectrums = []
    for segment in segments:
        correlation = np.correlate(segment, segment, mode='full')[segment_length-1:]    # np.correlate returns symetrical data. We only need one set. Trim off the first half.
        spectrum = np.abs(np.fft.rfft(correlation))
        # spectrum = np.abs(np.fft.rfft(segment))
        spectrums.append(spectrum)

    # Convert to numpy array
    spectrums = np.array(spectrums)

    # Calculate frequency and time axes
    freqs = np.fft.rfftfreq(segment_length, d=seconds_per_sample)
    times = df['Timestamp'].iloc[:-segment_length:next_segment_step_size]+ pd.Timedelta(seconds=(segment_length*seconds_per_sample)/2)

    # Plot
    # 1: Spectrograph (pcolormesh)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))  # 2 rows, 1 column
    im1 = ax1.pcolormesh(
        times, freqs, spectrums.T,
        shading='nearest'       # shows spectogram resolution better
        # shading='gouraud'     # smoother and prettier
    )
    fig.colorbar(im1, ax=ax1, label='Magnitude')
    ax1.set_ylabel('Frequency (Hz)')
    ax1.set_xlabel(f'Time, Centered on segment\n{segment_length} samples per segment | {segment_length*seconds_per_sample} seconds per segment | {int(segment_overlap_pct*100)}% overlap')
    ax1.set_ylim(0, 0.02)
    # ax1.set_title(f'EAC Spectrograph{" — " + title_ending if title_ending else ""} — segment_length: {segment_length}')
    ax1.set_title(f'EAC Spectrograph{" — " + title_ending if title_ending else ""}')

    # 2: Scatter and time visualizations
    ax2.scatter(df['Timestamp'], df[column_name], s=1)
    # ax2.set_title(f'Raw Data{" — " + title_ending if title_ending else ""} — samples: {len(data)}')
    ax2.set_title(f'Raw Data{" — " + title_ending if title_ending else ""}')
    ax2.set_ylabel(column_name)
    ax2.set_xlabel('Time')

    add_eclipse_and_flight_progress_highlights(df, ax2)



    # write to file
    os.makedirs('charts', exist_ok=True)
    plt.savefig(f"charts/{filename}")
    plt.close()
    return


# Call renderer for select balloons and columns
columns = [
    "Acceleration (g)",
    # "Battery Voltage",
    # "Light (lux)",
    # "IR (scale)",
    # "UVA (W/m^2)",
    "Temperature (F)",
    # "TempImputed",
    "WindSpeed (m/s)"
]
for balloon_name, balloon_df in df_interpolated.groupby('Balloon'):
    print(balloon_name)
    for column_name in columns:
        generate_spectrograph_and_scatterplot(
            balloon_df,
            column_name,
            f"{balloon_name} — {column_name}",
            f"Spectrograph_{balloon_name}_{column_name.replace("/","_")}.png"
        )
