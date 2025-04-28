import pandas as pd
import matplotlib as matplotlib
import numpy as np
import os

from scipy.io import wavfile
# from concurrent.futures import ThreadPoolExecutor

# Config
matplotlib.use('agg')


# Load and clean input data ---------------------------------------------------
df = pd.read_csv("eclipseData.csv", low_memory=False)

# PercentTotality has the string "Out of Path" mixed into some of it's data. (See Balloon == "Eclipse Pod v2 - PFW Eng")
# The mixed data causes Pandas to see the entire column as an object, and not as a numeric, messing up graph rendering and sorting.
df.loc[:, "PercentTotality"] = pd.to_numeric(df["PercentTotality"], errors="coerce").fillna(-1)

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

# Group by balloon, enforce timestamps to actualy arrive exactly every 2 seconds
# Fill in data for created timestamps using linear interpolation.
# Remove original data that does not line up on the 2 second intervals
# Merge interpolated data back together
def interpolate_balloons_to_two_second_intervals(df):

    # Ensure 'Timestamp' is of type datetime. Used for pd.date_range later
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Ensure timestamps are sorted for each balloon
    df = df.sort_values(['Balloon', 'Timestamp'])

    interpolated_balloon_dfs = []
    for balloon_name, balloon_df in df.groupby('Balloon'):
        # Create timestamp range with 2 second intervals
        start_time = balloon_df['Timestamp'].min()
        end_time = balloon_df['Timestamp'].max()
        complete_timestamps = pd.date_range(start=start_time, end=end_time, freq='2s')

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

        # Interpolate original data into new timestamps.
        # TODO: ⚠️ Check Summer Team's paper. Are there better interpolation methods for specific columns?
        merged_df = merged_df.interpolate(method='linear')

        # Purge original data that does not line up with complete_timestamps
        # This does mean we're sacrificing original data, but said data is accounted for during interpolation, and
        # this process ensures all timestamps (per balloon) are in 2 second intervals.
        merged_df = merged_df[merged_df['Timestamp'].isin(complete_timestamps)]

        # Save balloon's data.
        interpolated_balloon_dfs.append(merged_df)

    # Combine all balloons back together
    df_interpolated = pd.concat(interpolated_balloon_dfs, ignore_index=True)
    return df_interpolated

df_interpolated = interpolate_balloons_to_two_second_intervals(df_with_wind)

# Normalize each column to the WAV file format (-32768 to 32767)
def normalize_to_wav(data):
    return ((data - data.min()) * (65535) / (data.max() - data.min()) - 32768).astype(np.int16)

for balloon_name, balloon_df in df_interpolated.groupby('Balloon'):

    wav_normalized_data = normalize_to_wav(balloon_df["Temperature (F)"].fillna(0))

    os.makedirs('wav_outputs', exist_ok=True)
    wav_sample_rate = 44100
    # ↑ Our real data is at 0.5hz (1 sample per 2 seconds).
    # Any interesting data found from processing the audio data, devide frequency by (44100 * 2)
    wavfile.write(f'wav_outputs/{balloon_name}_temp.wav', wav_sample_rate, wav_normalized_data)
