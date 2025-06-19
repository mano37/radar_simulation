import numpy as np
from radarsimpy import Radar, Transmitter, Receiver
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.graph_objs as go
from IPython.display import Image
from plotly.subplots import make_subplots

pio.renderers.default = 'browser'  # 또는 'chrome', 'firefox' 등 명시 가능

############################################################
# Define Antenna patterns
############################################################

# Azimuth
az_angle = np.arange(-80, 81, 1)
az_pattern = 20 * np.log10(np.cos(az_angle / 180 * np.pi) ** 4) + 6

# Elevation
el_angle = np.arange(-80, 81, 1)
el_pattern = 20 * np.log10((np.cos(el_angle / 180 * np.pi)) ** 20) + 6

############################################################
# Define transmitter and receiver
############################################################

# Define transmitter channel

tx_channel = dict(
    location=(0, 0, 0),
    azimuth_angle=az_angle,
    azimuth_pattern=az_pattern,
    elevation_angle=el_angle,
    elevation_pattern=el_pattern,
)

# Define radar transmitter
tx = Transmitter(
    f=[24.075e9, 24.175e9],
    t=80e-6,
    tx_power=10,
    prp=100e-6,
    pulses=256,
    channels=[tx_channel],
)

# Define receiver channel
rx_channel = dict(
    location=(0, 0, 0),
    azimuth_angle=az_angle,
    azimuth_pattern=az_pattern,
    elevation_angle=el_angle,
    elevation_pattern=el_pattern,
)

# Define radar receiver
### fs - Sampling rate (sps)
### noise_figure - Noise figure (dB)
### rf_gain - Total RF gain (dB)
### load_resistor - Load resisstor to convert power to voltage (Ohm)
### baseband_gain - Total baseband gain (dB)

rx = Receiver(
    fs=2e6,
    noise_figure=12,
    rf_gain=20,
    load_resistor=50,
    baseband_gain=30,
    channels=[rx_channel],
)

radar = Radar(transmitter=tx, receiver=rx)

############################################################
# Define targets
############################################################
target_1 = dict(location=(200, 0, 0), speed=(-5, 0, 0), rcs=20, phase=0)
target_2 = dict(location=(150, 0, 0), speed=(-25, 0, 0), rcs=15, phase=0)
target_3 = dict(location=(50, 0, 0), speed=(0, 0, 0), rcs=5, phase=0)

targets = [target_2, target_3]


############################################################
# Simulate Baseband Signals
############################################################

from radarsimpy.simulator import sim_radar

data = sim_radar(radar, targets)

############################################################
# ITU Rainfall Attenuation Simulation (before sim_radar)
############################################################

rain_sim = True
def itu_rain_attenuation(frequency_ghz, rain_rate_mm_per_hr, path_length_km):
    """
    ITU-R P.838-3 모델에 따른 감쇠(dB) 계산
    """
    k = 0.22
    alpha = 0.88
    gamma_r = k * (rain_rate_mm_per_hr ** alpha)  # dB/km
    return gamma_r * path_length_km  # 왕복 감쇠량 [dB]

# 설정값
rain_rate = 100  # mm/hr
frequency = 24.125  # GHz
target_pos = np.array(targets[0]["location"])  # 하나만 있다고 가정

# 거리 및 감쇠 계산
range_km = np.linalg.norm(target_pos) * 2 / 1000  # 왕복 거리 [km]
loss_dB = itu_rain_attenuation(frequency, rain_rate, range_km)
attenuation_factor = 10 ** (-loss_dB / 20)  # 전압 감쇠계수

if rain_sim:
# baseband 전체에 감쇠 적용
    data["baseband"] *= attenuation_factor

timestamp = data["timestamp"]
baseband = data["baseband"] + data["noise"]


############################################################
# Radar Signal Processing
############################################################

# Range profile

from scipy import signal
import radarsimpy.processing as proc

range_window = signal.windows.chebwin(radar.sample_prop["samples_per_pulse"], at=60)
range_profile = proc.range_fft(baseband, range_window)
max_range = (
    3e8
    * radar.radar_prop["receiver"].bb_prop["fs"]
    * radar.radar_prop["transmitter"].waveform_prop["pulse_length"]
    / radar.radar_prop["transmitter"].waveform_prop["bandwidth"]
    / 2
)

range_axis = np.linspace(
    0, max_range, radar.sample_prop["samples_per_pulse"], endpoint=False
)

doppler_axis = np.linspace(
    0,
    radar.radar_prop["transmitter"].waveform_prop["pulses"],
    radar.radar_prop["transmitter"].waveform_prop["pulses"],
    endpoint=False,
)

fig = go.Figure()

fig.add_trace(
    go.Surface(
        x=range_axis,
        y=doppler_axis,
        z=20 * np.log10(np.abs(range_profile[0, :, :])),
        colorscale="Rainbow",
    )
)

fig.update_layout(
    title="Range Profile",
    height=600,
    scene=dict(
        xaxis=dict(title="Range (m)"),
        yaxis=dict(title="Chirp"),
        zaxis=dict(title="Amplitude (dB)"),
        aspectmode="cube",
    ),
)

# uncomment this to display interactive plot
fig.show()


############################################################
# Range-Doppler Processing
############################################################

doppler_window = signal.windows.chebwin(
    radar.radar_prop["transmitter"].waveform_prop["pulses"], at=60
)
range_doppler = proc.doppler_fft(range_profile, doppler_window)

unambiguous_speed = (
    3e8 / radar.radar_prop["transmitter"].waveform_prop["prp"][0] / 24.125e9 / 2
)

range_axis = np.linspace(
    0, max_range, radar.sample_prop["samples_per_pulse"], endpoint=False
)

doppler_axis = np.linspace(
    -unambiguous_speed,
    0,
    radar.radar_prop["transmitter"].waveform_prop["pulses"],
    endpoint=False,
)

unambiguous_speed = (
    3e8 / radar.radar_prop["transmitter"].waveform_prop["prp"][0] / 24.125e9 / 2
)

range_axis = np.linspace(
    0, max_range, radar.sample_prop["samples_per_pulse"], endpoint=False
)

doppler_axis = np.linspace(
    -unambiguous_speed,
    0,
    radar.radar_prop["transmitter"].waveform_prop["pulses"],
    endpoint=False,
)

fig = go.Figure()
fig.add_trace(
    go.Surface(
        x=range_axis,
        y=doppler_axis,
        z=20 * np.log10(np.abs(range_doppler[0, :, :])),
        colorscale="Rainbow",
    )
)

fig.update_layout(
    title="Range Doppler",
    height=600,
    scene=dict(
        xaxis=dict(title="Range (m)"),
        yaxis=dict(title="Velocity (m/s)"),
        zaxis=dict(title="Amplitude (dB)"),
        aspectmode="cube",
    ),
)

# uncomment this to display interactive plot
fig.show()
