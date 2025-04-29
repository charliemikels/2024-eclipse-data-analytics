import seaborn as sns
import matplotlib as matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from pandas import *
import numpy as np
from scipy.stats import shapiro
from scipy.stats import levene
import scipy.signal as signal
import folium
import csv
import pywt

###Select which data file you are working with:
data = read_csv("eclipseData.csv") ##Original Data
#data = read_csv("interpolated.csv") ##Interpolated Data
#data = read_csv("interpolated_with_wind.csv") ##Interpolated Data With Wind




###Data for each balloon. Based on Original Data
#Full flights of each balloon
AMADuring = [2, 3455, "AMADuring"]
AMAPre = [3456, 4852, "AMAPre"]
Eastbrook = [4853, 9736, "Eastbrook"]
GRCC = [9737, 13388, "B: GRCC"]
IMSDuring = [13389, 15126, "B: IMSDuring"]
IMSPre = [15127, 17568, "IMSPre"]
INAcademy = [17569, 17920, "INAcademy"]
LintonS = [17921, 18349, "LintonS"]
NSEUpland = [18350, 20274, "NSEUpland"]
PFWEdu = [20275, 22166, "PFWEdu"]
PFWEng = [22167, 25708, "PFWEng"]
Stockbridge = [25709, 27996, "E: Stockbridge"]
TaylorU = [27997, 29569, "TaylorU"]
USI = [29570, 32308, "USI"]

#Specialized subsets of balloon data
IMSDuringAscent = [13389, 14451, "IMSDuringAscent"]
IMSPreAscent = [15127, 16033, "IMSPreAscent"]
IMSPre50min = [15523, 16292, "IMSPre 50 Minutes"]
IMSDuring50min = [13890, 14620, "IMSDuring 50 Minutes"]

EastbrookAfterEclipse = [5590, 9736, "A: Eastbrook Post-Eclipse"]
GRCCAfterEclipse = [10419, 13388, "B: GRCC Post-Eclipse"]
IMSDuringAfterEclipse = [4269, 15126, "C: IMSDuring Post-Eclipse"]
NSEUplandAfterEclipse = [19110, 20274, "D: NSEUpland Post-Eclipse"]
PFWEduAfterEclipse = [21111, 22166, "E: PFWEdu Post-Eclipse"]
StockbridgeAfterEclipse = [26969, 27996, "F: Stockbridge Post-Eclipse"]

#Balloons with data only above 50,000ft elevation
Eastbrook50k = [5667, 6326, "Eastbrook +50k"]
IMSDuring50k = [14091, 14523, "IMSDuring +50k"]
IMSPre50k = [15721, 16127, "IMSPre +50k"]
NSEUpland50k = [19046, 19603, "NSEUpland +50k"]
PFWEdu50k = [21089, 21695, "PFWEdu +50k"]
PFWEng50k = [23707, 24971, "PFWEng +50k"]
TaylorU50k = [28680, 29041, "TaylorU +50k"]


#Balloons with data only above 40,000ft elevation.
IMSPre40k = [15655, 16193, "IMSPre +40k"] #12:43:27 -3:19:20 36mins
IMSDuring40k = [13994, 14576, "IMSDuring +40k"] #2:49:16 - 3:29:20 40mins
Eastbrook40k = [5568, 6337, "Eastbrook +40k"] #3:06:10-4:13:32 TOO FAR AFTER ECLIPSE?
NSEUpland40k = [18934, 19660, "NSEUpland +40k"] #2:57:46 3:51:36
PFWEdu40k = [20937, 21776, "PFWEdu +40k"] #2:58:21 - 3:57:55

#Balloons with data from 3:00 - 3:30
Eastbrook30Min = [5480, 5832, "Eastbrook 30 Minutes"]
GRCC30Min = [10278, 10660, "GRCC 30 Minutes"]
IMSDuring30Min = [14159, 14582, "IMS During 30 Minutes"]
NSEUpland30Min = [18958, 19405, "NSE Upland 30 Minutes"]
PFWEdu30Min = [20957, 21308, "PFW Edu 30 Minutes"]
Stockbridge30Min = [26798, 27325, "Stockbridge 30 Minutes"]

EastbrookShort = [4853, 7000, "A: Eastbrook"]

#Balloons with data from 3:00 - 4:00
Eastbrook60Min = [5480, 6205, "Eastbrook 60 Minutes"]
GRCC60Min = [10278, 11145, "GRCC 60 Minutes"]
IMSDuring60Min = [14159, 15126, "IMS During 60 Minutes"]
NSEUpland60Min = [18958, 19790, "NSE Upland 60 Minutes"]
PFWEdu60Min = [20957, 21803, "PFW Edu 60 Minutes"]
Stockbridge60Min = [26798, 27842, "Stockbridge 60 Minutes"]

#Subset of balloon data from 3:00 (and above 36,000 ft) until the balloon drops below 36,000 ft (lower stratosphere).
EastbrookSubset = [5480, 6436, "Eastbrook Subset"] #3:00 - 4:39:44
IMSDuringSubset = [14159, 14598, "IMS During Subset"] #3:00 - 3:30 54
INAcademySubset = [17734, 17799, "INAcademy Subset"] # 3:08:28 - 3:38:48 (Very large gaps of time in data. DO NOT USE)
LintonSSubset = [18207, 18349, "LintonS Subset"] # 3:00 - 3:40:45 Altitude not accurate (end of balloon data is at 75,000 ft,)
NSEUplandSubset = [18958, 19702, "NSE Upland Subset"] #3:00 - 3:53:44
PFWEduSubset = [20957, 21803, "PFW Edu Subset"] #3:00 - 3:59:24
PFWEngSubset = [22976, 25141, "PFW Eng Subset"] #3:00 - 5:11:21
StockbridgeSubset = [26798, 27296, "Stockbridge Subset"] #3:00 - 3:28:27
TaylorUSubset = [28569, 29129, "Taylor U Subset"] #3:21:40 - 4:02:29


###FOR INTERPOLATED DATA ONLY

#Full Balloon flights
EastbrookInterpolated = [16823, 25658, 'Eastbrook Interpolated']
GRCCInterpolated = [25660, 32481, 'GRCC Interpolated']
IMSDuringInterpolated = [32483, 35622, 'IMS During Interpolated']
NSEUplandInterpolated = [48208, 52784, 'NSE Upland Interpolated']
PFWEduInterpolated = [52786, 56486, 'PFW Edu Interpolated']
StockbridgeInterpolated = [63005, 69240, 'Stockbridge Interpolated']
INAcademyInterpolated = [43081, 45624, "INAcademy Interpolated"]
LintonSInterpolated = [45627, 48206, "LintonS Interpolated"]
PFWEngInterpolated = [56489, 63003, "PFWEng Interpolated"]
TaylorUInterpolated = [69243, 74675, "TaylorU Interpolated"]
USIInterpolated = [74678, 80545, "USI Interpolated"]


#Balloon flights from 3:00 PM until the balloon drops below 36,000 ft (end of lower stratosphere)
EastbrookInterpolatedSubset = [17857, 20855, 'Eastbrook Interpolated'] #3:00 - 4:39:56
GRCCInterpolatedSubset = [26923, 28063, 'GRCC Interpolated'] #NEVER REACHED 36,000ft
IMSDuringInterpolatedSubset = [33840, 34772, 'IMS During Interpolated'] #3:00 - 3:31:05
INAcademyInterpolatedSubset = [43993, 45294, "INAcademy Interpolated"] # 3:00 - 3:43:22
LintonSInterpolatedSubset = [46984, 48206, "LintonS Interpolated"] # 3:00 - 3:40:45 Altitude not accurate (end of balloon data)
NSEUplandInterpolatedSubset = [49269, 50887, 'NSE Upland Interpolated'] #3:00 - 3:53:56
PFWEduInterpolatedSubset = [54017, 55812, 'PFW Edu Interpolated'] #3:00 - 3:59:50
PFWEngInterpolatedSubset = [58070, 62010, "PFWEng Interpolated"] #3:00 - 5:11:21
StockbridgeInterpolatedSubset = [67209, 68062, 'Stockbridge Interpolated'] #3:00 - 3:28:26
TaylorUInterpolatedSubset = [72717, 73951, "TaylorU Interpolated"] #3:21:31 - 4:02:39 (Does not reach +36,000ft at 3:00)
USIInterpolatedSubset = [74678, 80545, "USI Interpolated"] #No altitude data.

#Balloon flights above 36,000 ft (lower stratosphere)
EastbrookInterpolatedStratosphere = [17930, 20855, 'Eastbrook Interpolated'] #3:02:26 - 4:39:56
IMSDuringInterpolatedStratosphere = [33440, 34772, 'IMS During'] #2:46:41 - 3:31:05
INAcademyInterpolatedStratosphere = [44070, 45294, "INAcademy Interpolated"] # 3:02:34 - 3:43:22
LintonSInterpolatedStratosphere = [46756, 48206, "LintonS Interpolated"] # 2:52:25 - 3:40:45 Altitude not accurate (end of balloon data)
NSEUplandInterpolatedStratosphere = [49130, 50887, 'NSE Upland Interpolated'] #2:55:22 - 3:53:56
PFWEduInterpolatedStratosphere = [53876, 55812, 'PFW Edu Interpolated'] #2:55:18 - 3:59:50
PFWEngInterpolatedStratosphere = [58666, 62010, "PFWEng Interpolated"] #3:19:53 - 5:11:21
StockbridgeInterpolatedStratosphere = [64500, 68062, 'Stockbridge Interpolated'] #1:29:42 - 3:28:26
TaylorUInterpolatedStratosphere = [72717, 73951, "TaylorU Interpolated"] #3:21:31 - 4:02:39 (Does not reach +36,000ft at 3:00)




#Balloon flights from 3:00-3:30 PM.
EastbrookInterpolatedSubset30Mins = [17857, 18757, 'Eastbrook Interpolated 30 Mins'] #3:00 - 3:30
IMSDuringInterpolatedSubset30Mins = [33840, 34740, 'IMS During Interpolated 30 Mins'] #3:00 - 3:30
NSEUplandInterpolatedSubset30Mins = [49269, 50169, 'NSE Upland Interpolated 30 Mins'] #3:00 - 3:30
PFWEduInterpolatedSubset30Mins = [54017, 54917, 'PFW Edu Interpolated 30 Mins'] #3:00 - 3:30
StockbridgeInterpolatedSubset30Mins = [67209, 68062, 'Stockbridge Interpolated 30 Mins'] #3:00 - 3:28:26

IMSDuringInterpolatedSubset36k = [33440, 34772, 'IMS During +36,000 ft'] #2:46:41 - 3:31:05
IMSPreInterpolatedSubset36k = [36421, 37647, 'IMS Pre +36,000 ft']




#This function gets data from specified balloons and calls functions.
#This is used specifically for the following three functions
def getBalloonData (Name):
    #Get specific data and put into ariable names
    altitude = data['AltImputed'][Name[0]:Name[1]].tolist()
    temperature = data['TempImputed'][Name[0]:Name[1]].tolist()
    acceleration = data['AccelImputed'][Name[0]:Name[1]].tolist()
    time = data['Timestamp'][Name[0]:Name[1]].tolist()
    print(Name[2])

    #Call functions with specific data.
    tempVsAltPlot(altitude, temperature, time, Name[2])



#This function graphs temperature and altitude on the same graph
def tempVsAltPlot(altitude, temperature, time, name):
    time2 = []
    for date_string in time:
        datetime_object = datetime.fromisoformat(date_string.replace("Z", "+00:00"))
        time2.append(datetime_object)

    x = time2
    y1 = altitude
    y2 = temperature
    x.pop()
    y1.pop()
    y2.pop()
    # Create the figure and primary y-axis
    fig, ax1 = plt.subplots()

    ###Sets the starting and ending times:
    start_time = datetime(2024, 4, 8, 14, 00)
    end_time = datetime(2024, 4, 8, 17, 00)
    plt.xlim(start_time, end_time)

    # Plot the first dataset
    color = 'tab:red'
    ax1.set_xlabel('time')
    ax1.set_ylabel('altitude (ft)', color=color)
    ax1.plot(x, y1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Create the secondary y-axis
    ax2 = ax1.twinx()

    # Plot the second dataset
    color = 'tab:blue'
    ax2.set_ylabel('temperature (F)', color=color)
    ax2.plot(x, y2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

    plt.suptitle(name)
    plt.show()


#Shapiro test function
def shapiroTest(balloonName):
    temperature = data['TempImputed'][balloonName[0]:balloonName[1]].tolist()
    tempData = np.array(temperature)
    shapiroTest = shapiro(tempData)
    print("Shapiro Test: ", shapiroTest)

#Variance calculation function
def varianceCalculation(balloonName):
    temperature = data['TempImputed'][balloonName[0]:balloonName[1]].tolist()
    tempData = np.array(temperature)
    variance = np.var(tempData)
    print("Variance:", variance)

#Levene's test function
def levenesTest(balloonName1, balloonName2):
    temperature1 = data['TempImputed'][balloonName1[0]:balloonName1[1]].tolist()
    temperature2 = data['TempImputed'][balloonName2[0]:balloonName2[1]].tolist()
    leveneTest = levene(temperature1, temperature2)
    print(balloonName1[2]," / ",balloonName2[2]," Levene's Test: ",leveneTest)
    print(len(temperature1))
    print(len(temperature2))

    df = DataFrame({
        'Temperature (F)': np.concatenate([temperature1, temperature2]),
        'Balloon': ['IMS Pre'] * len(temperature1) + ['IMS During'] * len(temperature2)
    })
    sns.boxplot(x="Balloon", y='Temperature (F)', data=df, showmeans=True)
    plt.title('IMS Pre and IMS During Temperature Distribution')
    plt.show()





##These are the calls to getBalloonData, where we can generate charts and run tests.
##
##getBalloonData(IMSDuring50min)
##getBalloonData(IMSPre50min)
##getBalloonData(AMADuring) #BAD ALTITUDE DATA
##getBalloonData(AMAPre)
##getBalloonData(Eastbrook)
##getBalloonData(GRCC)
##getBalloonData(IMSDuring)
##getBalloonData(IMSPre)
##getBalloonData(INAcademy)
##getBalloonData(LintonS)
##getBalloonData(NSEUpland)
##getBalloonData(PFWEdu)
##getBalloonData(PFWEng)
##getBalloonData(Stockbridge)
##getBalloonData(TaylorU)
##getBalloonData(USI)

##getBalloonData(IMSPreAscent)
##getBalloonData(IMSDuringAscent)

##getBalloonData(IMSPre40k)
##getBalloonData(IMSDuring40k)

##getBalloonData(IMSDuring50k)
##getBalloonData(IMSPre50k)
##getBalloonData(NSEUpland50k)
##getBalloonData(PFWEdu50k)
##getBalloonData(TaylorU50k)

##getBalloonData(Eastbrook30Min)
##getBalloonData(GRCC30Min)
###getBalloonData(IMSDuring30Min)
##getBalloonData(NSEUpland30Min)
##getBalloonData(PFWEdu30Min)
##getBalloonData(Stockbridge30Min)

##
#evenesTest(IMSPre40k, IMSDuring40k)
##levenesTest(IMSPre50k, NSEUpland50k)
##levenesTest(IMSPre50k, PFWEdu50k)
##levenesTest(IMSPre50k, TaylorU50k)

##getBalloonData(EastbrookInterpolated)
##getBalloonData(GRCCInterpolated)
##getBalloonData(IMSDuringInterpolated)
##getBalloonData(NSEUplandInterpolated)
##getBalloonData(PFWEduInterpolated)
##getBalloonData(StockbridgeInterpolated)


##getBalloonData(IMSDuringInterpolatedSubset36k)
##getBalloonData(IMSPreInterpolatedSubset36k)





#This function graphs multiple balloons' temperatures and compares them in the same graph.
def temperatureVsTime(balloonName1):#, balloonName2, balloonName3):
    #balloonNames = [balloonName1, balloonName2, balloonName3]
    balloonNames = [balloonName1]
    for balloonName in balloonNames:
        time = data['Timestamp'][balloonName[0]:balloonName[1]].tolist()
        temperature = data['TempImputed'][balloonName[0]:balloonName[1]].tolist()
        #temperature = data['WindSpeed (m/s)'][balloonName[0]:balloonName[1]].tolist() #Wind Speed
        time2 = []
        for date_string in time:
            datetime_object = datetime.fromisoformat(date_string.replace("Z", "+00:00"))
            time2.append(datetime_object)
        x1_values = time2
        y1_values = temperature
        x1_values.pop()
        y1_values.pop()
        plt.plot(x1_values, y1_values, label = balloonName[2])

    # Set x-axis limits
    start_time = datetime(2024, 4, 8, 14, 45)
    end_time = datetime(2024, 4, 8, 15, 30)
    plt.xlim(start_time, end_time)

    # Format x-axis to show HH:MM
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

    # Set y-axis limits
    plt.ylim(-65, -30)  # Adjust as needed

    # Add labels and title
    plt.xlabel("Time")
    plt.ylabel("Temperature (F)")
    #plt.ylabel("Wind Speed (m/s)")
    plt.title(balloonName1[2])#+" vs "+balloonName2[2]+" vs "+balloonName3[2])
    #leg = plt.legend(loc='upper center')
    # Display the graph
    plt.show()

#temperatureVsTime(IMSDuring, Eastbrook, GRCC)
#temperatureVsTime(IMSDuringInterpolatedStratosphere)




def normalizeData(dataset):
    maxValue = dataset.max()
    minValue = dataset.min()
    normalizedDataset = []
    for i in dataset:
        normalizedDataset.append((i-minValue)/(maxValue-minValue))
    normalizedDataset = np.array(normalizedDataset)
    return normalizedDataset



def coherenceSpectrum(fft1, fft2, balloonName):
    fs = 0.5
    f, Cxy = signal.coherence(fft1, fft2, fs)
    plt.plot(f, Cxy)
    plt.xlim(0.001, 0.01)
    plt.ylim(0, 1)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Coherence')
    plt.title('Coherence Spectrum (Temperature vs Wind Speed)')
    plt.suptitle(balloonName[2], fontsize=14)
    plt.grid(True)
    plt.show()


#This function applies a butterworth filter to the temperature data of a balloon.
    #If you want to switch the data being evaluated, change what data it calls in the code and respective variable names.
def butterworth_filter(balloonName, order=2):
    """Applies a Butterworth low-pass filter to data.

    Args:
        data: NumPy array of data.
        cutoff_frequency: Cutoff frequency (Hz).
        sampling_rate: Sampling rate (Hz).
        order: Filter order.

    Returns:
        Filtered data.
    """
    #tempData = data['WindSpeed (m/s)'][balloonName[0]:balloonName[1]].tolist() #Wind Speed
    tempData = data['TempImputed'][balloonName[0]:balloonName[1]].tolist()
    tempData = np.array(tempData)

    cutoff_frequency = 0.1
    b, a = signal.butter(order, cutoff_frequency, btype='low', analog=False)
    filtered_data = signal.lfilter(b, a, tempData)
    filtered_data = filtered_data[10:]##removing the first 10 entries because it is very skewed in butterworth
    #Plotting filtered data vs normal data
    time = np.arange(len(tempData)) / 2
    time = time[10:] ##removing the first 10 entries because it is very skewed in butterworth
    tempData = tempData[10:]##removing the first 10 entries because it is very skewed in butterworth

    ### Unhide the following code if you want to view a graph of the temperature data vs the butterworth filter.
##    plt.figure(figsize=(12, 6))
##    plt.plot(time, tempData, label='Original Data')
##    plt.plot(time, filtered_data, label='Filtered Data')
##    plt.xlabel('Time')
##    plt.ylabel('Temperature')
##    plt.legend()
##    plt.title(balloonName[2]+' Butterworth Low-Pass Filter')
##    plt.grid(True)
##    plt.show()


    return filtered_data


#This function applies a Fast Fourier Transform to the filtered temperature data. Using the NumPy Library
def fftNumPy(filtered_data, balloonName):

    n = len(filtered_data)
    frequencies = np.fft.fftfreq(n, d=2) #sample spacing (2 seconds)
    fft_values = np.fft.fft(filtered_data)
    amplitudes = np.abs(fft_values)  # Get the magnitude (amplitude)

    # Only keep positive frequencies (and corresponding amplitudes)
    positive_frequencies = frequencies[:n // 2]
    positive_amplitudes = amplitudes[:n // 2]

    plt.plot(positive_frequencies, positive_amplitudes)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    #plt.title(balloonName[2]+' Fourier Transform - Temperature Data')
    plt.title(balloonName[2]+' Fourier Transform - Wind Speed Data')
    plt.xlim(0.001, 0.01)
    plt.ylim(0, 4000)
    plt.grid(True)
    plt.show()

    return positive_frequencies, positive_amplitudes


#This function applies a Fast Fourier Transform to the filteres temperature data. Using code from https://pythonnumericalmethods.studentorg.berkeley.edu/notebooks/chapter24.04-FFT-in-Python.html
from numpy.fft import fft, ifft
def fftFunction(x, balloonName):
    sr = 0.5
    ts = 1.0/sr

    X = fft(x)
    N = len(X)
    n = np.arange(N)
    T = N/sr
    freq = n/T

    t = np.arange(0,2*N,ts)

    plt.figure(figsize = (12, 6))

    plt.stem(freq, np.abs(X), 'b', \
             markerfmt=" ", basefmt="-b")
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')
    plt.xlim(0.001, 0.01)
    plt.ylim(0,2500)
    plt.suptitle(balloonName[2], fontsize=14)
    plt.show()

#selections = [balloonNames]
##for name in selections:
##    filtered_data = butterworth_filter(name)
##    ###Pick which FFT implementation to use (Recommend fftFunction). Both seem to have about the same result,
##        #fftFunction graphs using more of a bar chart for visualization, whereas fftNumPy uses a line graph.
##    fftFunction(filtered_data,name)
##    #fftNumPy(filtered_data,name)
##





#This function plots the flight paths of IMS Pre and IMS During on a map application
def plot_lat_lon(balloonName, balloonName1,
                  label1="IMS Pre", label2="IMS During",
                  title="Latitude vs. Longitude",
                  xlabel="Longitude", ylabel="Latitude"):
    """
    Plots two datasets of latitude and longitude coordinates.

    Args:
        lat1: List or array of latitude values for the first dataset.
        lon1: List or array of longitude values for the first dataset.
        lat2: List or array of latitude values for the second dataset (optional).
        lon2: List or array of longitude values for the second dataset (optional).
        label1: Label for the first dataset in the plot legend.
        label2: Label for the second dataset in the plot legend.
        title: Title of the plot.
        xlabel: Label for the x-axis (longitude).
        ylabel: Label for the y-axis (latitude).
    """
    lat1 = data['AdjLatFilled'][balloonName[0]:balloonName[1]].tolist()
    lat1.pop()
    lat1 = np.array(lat1)
    lon1 = data['AdjLonFilled'][balloonName[0]:balloonName[1]].tolist()
    lon1.pop()
    lon1 = np.array(lon1)
    lat2 = data['AdjLatFilled'][balloonName1[0]:balloonName1[1]].tolist()
    lat2.pop()
    lat2 = np.array(lat2)
    lon2 = data['AdjLonFilled'][balloonName1[0]:balloonName1[1]].tolist()
    lon2.pop()
    lon2 = np.array(lon2)

    # Create a map centered at a specific location
    m = folium.Map(location=[40, -85.5], zoom_start=10)
    coordinates = []
    coordinates2 = []
    for i in range(len(lat1)):
        coordinates.append([lat1[i],lon1[i]])
        #folium.Marker([lat1[i], lon1[i]]).add_to(m)
    for i in range(len(lat2)):
        coordinates2.append([lat2[i],lon2[i]])
    folium.PolyLine(
        locations=coordinates,
        color="#0000FF",
        weight=5
    ).add_to(m)

    folium.PolyLine(
        locations=coordinates2,
        color="#00FF00",
        weight=5
    ).add_to(m)

    legend_html = '''
    <div style="position: fixed;
         bottom: 50px; right: 50px; width: 150px; height: 100px;
         border:2px solid grey; z-index:9999; font-size:16px;
         background-color:white; opacity: 0.85;">
         &nbsp; <b>Legend</b> <br>
         &nbsp; IMS Pre &nbsp; <i class="fa fa-circle" style="color:#0000FF"></i><br>
         &nbsp; IMS During &nbsp; <i class="fa fa-circle" style="color:#00FF00"></i><br>
    </div>
    '''

    # Add the legend to the map
    m.get_root().html.add_child(folium.Element(legend_html))

    # Save the map
    m.save('balloon.html')

#plot_lat_lon(IMSPre, IMSDuring)




###The next function is the linear interpolation code from Charlie.
###When this function is ran once, a new file with dataset will be created, and you must reference that file instead to work with interpolated data.
# Fix the few stray stings in "PercentTotality" to be numeric values.
    data.loc[:, "PercentTotality"] = to_numeric(data["PercentTotality"], errors="coerce").fillna(-1)

    # Ensure 'Timestamp' is of type datetime. Used for pd.date_range later
    df['Timestamp'] = to_datetime(df['Timestamp'])
# Group by balloon, enforce timestamps to actually arrive exactly every 2 seconds
# Fill in data for created timestamps using linear interpolation.
# Remove original data that does not line up on the 2 second intervals
# Merge interpolated data back together
def interpolate_balloons_to_two_second_intervals(df):

    # Ensure timestamps are sorted for each balloon
    df = df.sort_values(['Balloon', 'Timestamp'])

    interpolated_balloon_dfs = []
    for balloon_name, balloon_df in df.groupby('Balloon'):
        # Create timestamp range with 2 second intervals
        start_time = balloon_df['Timestamp'].min()
        end_time = balloon_df['Timestamp'].max()
        complete_timestamps = date_range(start=start_time, end=end_time, freq='2s')

        # Create new dataframe with complete timestamps
        populated_timestamps_df = DataFrame({'Timestamp': complete_timestamps})
        populated_timestamps_df['Balloon'] = balloon_name

        # Combine original data and new timestamps. Creates columns and nulls to match original data.
        merged_df = merge(
            left=populated_timestamps_df,
            right=balloon_df,
            on=['Timestamp', 'Balloon'],
            how='outer' # "Outer" so that the original data is still present.
                        # sometimes the 2 second timestamps don't line up, so we need the original data for interpolation.
        )

        # Interpolate original data into new timestamps.
        # TODO: Check Summer Team's paper. Are there better interpolation methods for specific columns?
        # TODO: Pandas is throwing warnings about interpolating object types. Some of the objects are strings
        merged_df = merged_df.interpolate(method='linear')

        # Purge original data that does not line up with complete_timestamps
        # This does mean we're sacrificing original data, but said data is accounted for during interpolation.
        # This step ensures all timestamps (per balloon) are in 2 second intervals.
        merged_df = merged_df[merged_df['Timestamp'].isin(complete_timestamps)]

        # Done with this balloon
        interpolated_balloon_dfs.append(merged_df)

    # Combine all balloons back together
    df_interpolated = concat(interpolated_balloon_dfs, ignore_index=True)
    return df_interpolated

#df_interpolated = interpolate_balloons_to_two_second_intervals(data)
##print(type(df_interpolated))
#df_interpolated.to_csv('interpolated.csv')
