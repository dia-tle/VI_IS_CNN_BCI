## MNE Preprocessing script for Biosemi EEG data


# Import the relevant libraries

import mne
import numpy as np
import matplotlib as qt



### 1. LOADING     ###

# Set the path to the BDF file


data_path = 'P00/P00_VI_---.bdf'


# Load the raw data
raw = mne.io.read_raw_bdf(data_path, preload=True)

raw.info['ch_names']


# Data matrix information row,column
raw.get_data().shape


# print channel names
print(raw.info)

print(raw.info['ch_names'])


# Find trigger (status) events

events = mne.find_events(raw, initial_event=True) 

#np.save('events.npy', events)   # save string of trigger values

# Plot the raw information 

# nExample*nChannel*nTime
# -1, 0, +4 = epoching 
# -1, 0 = baseline correction

raw.plot(n_channels=64)



### 2. UTILITIES ###

# Show channels 

print('list_channels: ', raw.info['ch_names'])
#print()
print('data_shape: ', raw.get_data().shape)
#print()

# Choose channels to keep - not all channels needed

channels_to_keep = ["Fp1","AF7","AF3","F1-0","F3-0","F5-0","F7-0","FT7","FC5","FC3","FC1","C1-0","C3-0","C5-0","T7","TP7","CP5","CP3","CP1","P1","P3","P5","P7","P9","PO7","PO3","O1","Iz","Oz","POz","Pz","CPz","Fpz","Fp2","AF8","AF4","AFz","Fz","F2-0","F4-0","F6-0","F8-0","FT8","FC6","FC4","FC2","FCz","Cz","C2-0","C4-0","C6-0","T8","TP8","CP6","CP4","CP2","P2","P4","P6","P8","P10","PO8","PO4","O2","EXG1","EXG2","EXG3","EXG4"]
#"EXG1","EXG2","EXG3","EXG4"

# Collect channels to drop by checking if the channel is NOT in channels_to_keep
channels_to_drop = [chan_name for chan_name in raw.ch_names if chan_name not in channels_to_keep]

# Drop the channels in one call after the loop
raw.drop_channels(channels_to_drop)

#for chan_name in raw.ch_names: 
   # if chan_name not in channels_to_keep: 
        #raw.drop_channels([chan_name])  
        
        
# The updated channel list
print(raw.info['ch_names'])


# Create a mapping to rename channels in raw to match biosemi_montage 

channel_mapping = {
    "F1-0": "F1", 
    "F3-0": "F3",
    "F5-0": "F5",
    "F7-0": "F7",
    "C1-0": "C1",
    "C3-0": "C3",
    "C5-0": "C5",
    "F2-0": "F2",
    "F4-0": "F4",
    "F6-0": "F6",
    "F8-0": "F8",
    "C2-0": "C2",
    "C4-0": "C4",
    "C6-0": "C6"
   }


# Rename the channels in 'raw'

raw.rename_channels(mapping=channel_mapping)


# Set channel types for EXG 

raw.set_channel_types({'EXG1': 'ecg', 
                       'EXG2': 'ecg', 
                       'EXG3': 'ecg', 
                       'EXG4': 'eog'}

 )



# Generate and apply montage 

biosemi_montage = mne.channels.make_standard_montage('biosemi64') 

raw.set_montage(biosemi_montage)



# Visualise raw continuous data 

n_channels = len(raw.info['ch_names'])
raw.plot(n_channels=n_channels, scalings=1e-5)

raw.compute_psd().plot() # plot power spectrum




## INTERPOLATION 
## Should be performed before Epoching

# Mark bad channels

#bad_channels = ['P3','Iz','POz','C6','CP2']
#raw.info['bads'] = bad_channels


# Interpolate the specified channel
#raw.interpolate_bads(reset_bads=True, verbose=True)

# Plot
#raw.plot(n_channels=68, scalings=1e-5)



### 3. FILTERING   ###

# Utilities before epoching

# Apply notch filter to remove power-line noise 50Hz 
filtered_data = raw.copy().notch_filter(np.arange(50, 200, 50))
filtered_data = filtered_data.filter(0.5, 125) 
# in CNN 7-30 alpha beta
# 70 or 0.5Hz - 125Hz per literature to keep high gamma

# Set EEG preference 
filtered_data, _ = mne.set_eeg_reference(filtered_data)


# Visualise again

n_channels = len(raw.info['ch_names'])
filtered_data.plot(n_channels=n_channels, scalings=1e-5)

filtered_data.compute_psd().plot()    # plot power spectrum

# Interpolate 

bad_channels = ['']
filtered_data.info['bads'] = bad_channels


# Interpolate the specified channel
filtered_data.interpolate_bads(reset_bads=True, verbose=True)

# Plot
filtered_data.plot(n_channels=68, scalings=1e-5)



### 4. EPOCHING   ###
# The Epochs data structure represents  segmented and extracted portions of interest from the raw data. 
# It allows you to focus on specific time intervals of the EEG recording, typically corresponding to specific events or experimental conditions
# The Epochs object is created by defining the time windows around event markers or triggers in the Raw data. 
# Creat an mne.Epochs object by extracting epochs around the event markers 
# Apply baseline correction to the epochs if necessary 


# Events - find triggers?


temp_events = events 
temp_events[:, 2] &= [2**7-1]

# use 7-1 in the rule instead of 9 

# save into a list to be consistent 
events = [temp_events]


print(events)


# Epoching 

event_id = {'VIrelax': 1, 'VIpush': 2, 'ISrelax': 4, 'ISpush': 5}  
# code VI: relax, push, IS: relax, push




epochs = mne.Epochs(raw=filtered_data, events=events[0], event_id=event_id, tmin=-0.5, tmax=4, preload=True, baseline=None)
# No baseline correction
# 0 bad epochs dropped


print(epochs.get_data().shape) 
# (196, 68, 4609) NEpochs,Nchannels,Ntimepoints
# nEpochs= nTrials


# Gives plot of Epochs which can then delete Epochs not included

epochs.plot(n_epochs=4, n_channels=64, scalings=1e-4)  


# After selecting epoch to drop (highlights in red) close window 
# It will automatically drop 

epochs.get_data().shape
# If deleted epochs, use to check dataframe after



###    5. ARTIFACT REJECTION & CLEANING    ### 


# 1. Run fast Independent Component Analysis (FastICA)

# import sklearn package ?

from mne.preprocessing import ICA 


n_components = epochs.get_data().shape[1]
ica = ICA(n_components=0.999999, max_iter=800, method='fastica', random_state=37)
ica.fit(epochs, decim=3)


# To fix the 1D array error 
# Output: All picks must be the range of the ICA components found - 63
picks = range(0,50)  



# Plot 
#ica.plot_components(epochs) - not used

ica.plot_components(picks)
# plots the head spectral ICA components

ica.plot_sources(epochs)
# Plots the components in raw continuous data




# Applying ICA to Epochs 

# Check components
ica.plot_properties(raw, picks=[0])

# Components to reject
components_to_reject = [0]    # 0 - eye sblinks, array of indices to reject

ica.exclude = components_to_reject 
# Exclude the ICA components 

ica.apply(epochs)
# Applying ICA to Epochs instance 
# Epochs 195 events (all good) -0.5 - 4 s, baseline off


# Plot epochs - clean
epochs.plot(n_epochs=4, n_channels=64)   


# Check shape if dropped epochs
epochs.get_data().shape
# nEpochs,nChannels,nTime


# RESAMPLING DATA to 256 Hz
# Save as one Epoch file 

epochs = epochs.resample(sfreq=256)

# Create a dataframe for epochs
# Epochs = epochs.get_data()
#(195,68,1152): nEpochs,nChannels,nTime, Array of float64


# Show shape
print(epochs.get_data().shape)
# (200,68, 1152)

# Save as epoch file

epochs.save('Epochs_00-epo.fif')



# Plot spectrum map 
spectrum = epochs.compute_psd()

spectrum.plot_topomap()



# NOT USED 


# Separate Epochs by paradigm and condition 
# Visual Imagery
#epochs_VIrelax = epochs['VIrelax'].resample(sfreq=256)

#epochs_VIpush = epochs['VIpush'].resample(sfreq=256)

# Imagined Speech
#epochs_ISrelax = epochs['ISrelax'].resample(sfreq=256)

#epochs_ISpush = epochs['ISpush'].resample(sfreq=256)


# Separate data between conditions 'push' and 'relax' into arrays

#data_VIrelax = epochs_VIrelax.get_data()
#data_VIpush = epochs_VIpush.get_data()

#data_ISrelax = epochs_ISrelax.get_data()
#data_ISpush = epochs_ISrelax.get_data()


# Show shape of data 
# nEpochs,nChannels,nTime

#data_VIrelax.shape
#data_VIpush.shape
#data_ISrelax.shape
#data_ISpush.shape



# Save the epochs in specific data frames ready for analysis 
# Dataframes nEpochs,nChannels,nTime 

# Using NumPy

# Save each data array to a seperate NumPy file for each condition
# Save data file per participant number

#np.save('data01_VIrelax.npy', data_VIrelax)
#np.save('data01_VIpush.npy', data_VIpush)
#np.save('data01_ISrelax.npy', data_ISrelax)
#np.save('data01_ISpush.npy', data_ISpush)

