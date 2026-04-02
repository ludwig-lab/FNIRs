import os
import mne
import numpy as np
import pandas as pd
import pyecap
from flexNIRs.fnirs_functions import *

import matplotlib.pyplot as plt
import h5py
import mne
import mne.channels
import mne.io.snirf
import mne_nirs.experimental_design
import mne_nirs.statistics

from nilearn.plotting import plot_design_matrix

import mne_nirs
from mne_nirs.channels import get_long_channels, get_short_channels, picks_pair_to_idx
from mne_nirs.experimental_design import make_first_level_design_matrix
from mne_nirs.statistics import run_glm

class kernel():

    def __init__(self, data_path, hash, snirf_type):
        fileLIST = os.listdir(data_path + r'\\SNIRF')
        snirf_file = [item for item in fileLIST if (hash.lower() in item.lower()) and (snirf_type.lower() in item.lower())][0]
        full_path = data_path + r'\\SNIRF\\' + snirf_file

        raw = mne.io.snirf.read_raw_snirf(full_path)
        raw.load_data()

        self.fs = raw.info['sfreq']
        self.d_type = snirf_type

        """Get Sync Signal"""
        f = h5py.File(full_path)
        sync_signal = f['nirs']['aux22']['dataTimeSeries'][:]
        sync_time = f['nirs']['aux22']['time'][:]
        stim_idx = np.nonzero(np.diff(np.squeeze(sync_signal)) ** 2 > 1)[0]

        idxLIST = []
        min_distance = 10
        for idx in stim_idx:
            if all(abs(idx - valid_idx) >= min_distance for valid_idx in idxLIST):
                idxLIST.append(idx)
        stim_idx = idxLIST
        self.stim_times = list(zip(sync_time[stim_idx[::2]], sync_time[stim_idx[1::2]]))

        """Get Kernel Data"""
        self.d = raw[:][0]  # Channel data in numpy array
        self.time = raw[:][1]  # Time values in numpy array

        probe_keys = [
            ("detectorLabels", str),
            ("sourceLabels", str),
            ("sourcePos3D", float),
            ("detectorPos3D", float),
        ]
        with h5py.File(full_path, "r") as file:
            probe_data = {
                key: np.array(file["nirs"]["probe"][key]).astype(dtype)
                for key, dtype in probe_keys
            }
        [*probe_data]

        idx_sources = np.array([int(ch.split("_")[0][1:]) - 1 for ch in raw.ch_names])
        idx_detectors = np.array([int(ch.split("_")[1].split(" ")[0][1:]) - 1 for ch in raw.ch_names])
        source_positions = np.array(probe_data["sourcePos3D"])[idx_sources]
        detector_positions = np.array(probe_data["detectorPos3D"])[idx_detectors]

        mods = [int(mod.split('M0')[1].split('S0')[0]) for mod in np.array(probe_data['sourceLabels'])[idx_sources]]

        sds = np.sqrt(np.sum((source_positions - detector_positions) ** 2, axis=1)).astype(int)

        if snirf_type == 'hb':
            wavelengths = [(ch.split(' ')[1]) for ch in raw.ch_names]
            d_type = wavelengths
        else:
            wavelengths = [int(ch.split(' ')[1]) for ch in raw.ch_names]
            d_type = [ch.split(' ')[2] for ch in raw.ch_names]

        """
        Construct DataFrame for Channel Indexing
        Columns: Channel name, Source, Detector, SDS, Wavelength, Moment (time bin for gate snirf)
        """
        self.channelDF = pd.DataFrame({'Channel': raw.ch_names,
                                  'Source': idx_sources + 1,
                                  'Detector': idx_detectors + 1,
                                  'Module': mods,
                                  'SDS': sds,
                                  'Wavelength': wavelengths,
                                  'Classifier': d_type, }
                                 )

        """For Hb Moments Data: Calculate HbT and add to data array and indexing dataframe"""
        if snirf_type.lower() == 'hb':
            sd_combos = list(dict.fromkeys([ch.split(' ')[0] for ch in raw.ch_names]))
            sd_pairs = [list(self.channelDF.index[self.channelDF['Channel'].str.contains(combo)]) for combo in sd_combos]

            HbT = []
            for pair in sd_pairs:
                total = np.sum(self.d[pair], axis=0)
                HbT.append(total)
            HbT = np.array(HbT)

            sd_idx = [idx[0] for idx in sd_pairs]
            HbT_DF = self.channelDF.loc[sd_idx]
            HbT_DF.reset_index(inplace=True, drop=True)
            HbT_DF['Channel'] = HbT_DF['Channel'].str.replace('hbo', 'hbt')
            HbT_DF['Wavelength'] = HbT_DF['Wavelength'].str.replace('hbo', 'hbt')
            HbT_DF['Classifier'] = HbT_DF['Wavelength']

            """Integrate data array and channelDF back into original"""
            self.d = np.vstack((self.d, HbT))
            self.channelDF = pd.concat([self.channelDF, HbT_DF], ignore_index=True)

    def get_data_indices(self, mod, wavelength, sds, classifier, s = None, d = None):
        # data_idx =
        if self.d_type == 'hb':
            pass
        pass

    def fir_filter(self, filter_cutoffs=(0.01, 0.2), transition_width=0.01, numtaps=301, freq_plot=False):
        """Setup Filter"""
        filter_weights = signal.firwin(numtaps, filter_cutoffs, width=transition_width, window='Hamming',
                                       pass_zero='bandpass', fs=self.fs)

        if freq_plot:
            w, h = signal.freqz(filter_weights, worN=fft.next_fast_len(40000, real=True))
            plt.plot((w / np.pi) * (self.fs / 2), 20 * np.log10(np.abs(h)))
            plt.xlim((0, 2))
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude (dB)')
            plt.show()
        #print('Filtering data with')
        self.filt_d = signal.filtfilt(filter_weights, [1], self.d, axis=1)