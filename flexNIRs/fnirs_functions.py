import h5py
import numpy as np
import pandas as pd
import scipy.io as sio
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
import matplotlib as mpl
from matplotlib import get_backend as mpl_get_backend
from scipy import signal
from scipy import fft
import plotly.graph_objs as go
from plotly.subplots import make_subplots

class flexNIRs():

    def \
            __init__(self, filePATH):
        flex_data = sio.loadmat(filePATH, struct_as_record = True)['data'][0,0]

        #Metadata
        self.file_path = filePATH
        self.fs = np.squeeze(flex_data['fs']).item()
        self.time = np.squeeze(flex_data['Time'])
        self.wv = flex_data['wv']
        self.sds = flex_data['sds']
        self.extinctionCoeff = flex_data['extinctionC']
        self.stimDF = None

        #Channels
        self.raw_channels = ['SS Red', 'SS IR',
                    'D1 Ambient', 'D1 LS Red', 'D1 LL Red', 'D1 LS IR', 'D1 LL IR',
                    'D3 Ambient', 'D3 LS Red', 'D3 LL Red', 'D3 LS IR', 'D3 LL IR']

        self.ch_names = ['SS Red', 'SS IR',
                         'D1 LS Red', 'D1 LL Red', 'D1 LS IR', 'D1 LL IR',
                         'D3 LS Red', 'D3 LL Red', 'D3 LS IR', 'D3 LL IR']

        self.ch_dct = {0 : ['SS', 'Red', 'HbO'],
                        1: ['SS', 'IR', 'HbR'],
                        2: ['D1 LS', 'Red', 'HbO'],
                        3: ['D1 LL', 'Red', 'HbO'],
                        4: ['D1 LS', 'IR', 'HbR'],
                        5: ['D1 LL', 'IR', 'HbR'],
                        6: ['D3 LS', 'Red', 'HbO'],
                        7: ['D3 LL', 'Red', 'HbO'],
                        8: ['D3 LS', 'IR', 'HbR'],
                        9: ['D3 LL', 'IR', 'HbR']            ,
                        }

        self.name_dct = { 'SS Red' : 0, 'SS IR' : 1,
                       'D1 LS Red' : 2, 'D1 LL Red' : 3, 'D1 LS IR' : 4, 'D1 LL IR' : 5,
                       'D3 LS Red' : 6, 'D3 LL Red' : 7, 'D3 LS IR' : 8, 'D3 LL IR' : 9
                       }

        self.ch_pairs = {'SS' : (0,1),
                         'D1 LS': (2,4),
                         'D1 LL' : (3,5),
                         'D3 LS' : (6,8),
                         'D3 LL' : (7,9)
                         }


        #Variable Assumptions from flexNIRs MATLAB (self_calibrated_fNIRs function)
        self.waterP = flex_data['waterP']
        self.msp = flex_data['msp']

        #Calculated coefficient variables from flexNIRs MATLAB (self_calibrated_fNIRs function)
        self.muaLIST = np.squeeze(flex_data['muaList'])
        self.est_dpfs = np.squeeze(flex_data['est_dpfs'])

        #Calculated coefficient variables from flexNIRs MATLAB (est_dpfs function)
        self.dpfs = np.squeeze(flex_data['dpfs'])

        #Raw data from flexNIRs MATLAB script.  Remove ambient channels and adjust dataframe to match others
        self.raw_data = pd.DataFrame(flex_data['raw_data'], columns = self.raw_channels)
        self.ambient_data = self.raw_data[['D1 Ambient', 'D3 Ambient']]
        self.raw_data.drop(columns = ['D1 Ambient', 'D3 Ambient'], inplace=True)

        #Unfiltered OD/Mua calculated by flexNIRs MATLAB script
        self.d_OD = pd.DataFrame(flex_data['deltaOD_raw'], columns = self.ch_names)
        self.d_Mua = pd.DataFrame(flex_data['deltaMua_raw'], columns = self.ch_names)

        #Filtered OD & Mua calculated by flexNIRs MATLAB Script
        self.d_OD_filt = pd.DataFrame(flex_data['deltaOD_filt'], columns = self.ch_names)
        self.d_Mua_filt = pd.DataFrame(flex_data['deltaMua_filt'], columns = self.ch_names)

        self.calc_hemoglobin()
        self.calc_hr(channel = 'SS Red')

    def manual_alignment(self, stimDF, stim_start_index):
        """Iterate through the stim dataframe and add stimulation start times to each row"""
        stim_index = stim_start_index
        stim_time = self.time[stim_index]

        """Get Indices of (approximate) times of all times and indices based on the first stim alignment"""
        fnirs_fs = self.fs
        fnirs_dt = 1 / fnirs_fs

        first_stim = stimDF.index[0][1]

        for stim_count, stim in enumerate(stimDF.index):

            # Sets up first stimulation data
            if stim[1] == first_stim:
                stimDF.loc[(0, first_stim), 'fNIRs onset index'] = stim_index
                stimDF.loc[(0, first_stim), 'fNIRs onset time (s)'] = stim_time

                stimDF.loc[(0, first_stim), 'fNIRs offset time (s)'] = stim_time + (
                        stimDF.loc[(0, first_stim), 'duration (ms)'] * 1e-3)

                stimDF.loc[(0, first_stim), 'fNIRs offset index'] = (
                            stimDF.loc[(0, first_stim), 'fNIRs offset time (s)'] / fnirs_dt).astype(int)
                stim_idx_span = stimDF.loc[(0, first_stim), 'fNIRs offset index'] - stimDF.loc[
                    (0, first_stim), 'fNIRs onset index']

            else:
                # Start of stimulation relative to first stimulation
                stim_start = stimDF.loc[stim, 'onset time (s)'] - stimDF.loc[(0, first_stim), 'onset time (s)']

                # Add time difference between current stim and first stim to fNIRs time
                stimDF.loc[stim, 'fNIRs onset time (s)'] = stimDF.loc[
                                                               (0, first_stim), 'fNIRs onset time (s)'] + stim_start
                stimDF.loc[stim, 'fNIRs onset index'] = (stimDF.loc[stim, 'fNIRs onset time (s)'] / fnirs_dt).astype(
                    int)

                # Add offset time based on duration of current stim
                stimDF.loc[stim, 'fNIRs offset time (s)'] = stimDF.loc[stim, 'fNIRs onset time (s)'] + (
                        stimDF.loc[stim, 'duration (ms)'] * 1e-3)

                """This ensures stimulation spans same # indices to account for discrepancies in sampling rates"""
                # stimDF.loc[stim, 'fNIRs offset index'] = (stimDF.loc[stim, 'fNIRs offset time (s)'] / fnirs_dt).astype(int)
                stimDF.loc[stim, 'fNIRs offset index'] = (stimDF.loc[stim, 'fNIRs onset index'] + stim_idx_span).astype(
                    int)
        #stimDF.reset_index(inplace=True)
        self.stimDF = stimDF

    def calc_hemoglobin(self):
        """Calculated HB changes per channel"""
        channel_pairs = {'SS': ['SS Red', 'SS IR'],
                         'D1 LS': ['D1 LS Red', 'D1 LS IR'],
                         'D1 LL': ['D1 LL Red', 'D1 LL IR'],
                         'D3 LS': ['D3 LS Red', 'D3 LS IR'],
                         'D3 LL': ['D3 LL Red', 'D3 LL IR'], }
        # channel_pairs = [(0,1),(2,4),(3,5),(6,8),(7,9)]
        exC = self.extinctionCoeff[:, 0:2]

        hemo_data = []
        hemo_data_f = []
        hemo_chanLIST = []

        for pair in channel_pairs:
            ch_red = channel_pairs[pair][0]
            ch_ir = channel_pairs[pair][1]

            #Unfiltered Mua values
            data_red = self.d_Mua[ch_red].values
            data_ir = self.d_Mua[ch_ir].values

            d = np.vstack((data_red, data_ir))
            Hb = np.matmul(exC ** -1, d)

            hemo_data.append(Hb[0, :])
            hemo_data.append(Hb[1, :])
            hemo_data.append(Hb[0, :] + Hb[1, :])

            #Filtered Mua Values
            data_red_f = self.d_Mua_filt[ch_red].values
            data_ir_f = self.d_Mua_filt[ch_ir].values
            d_f = np.vstack((data_red_f, data_ir_f))
            Hb_f = np.matmul(exC ** -1, d_f)

            hemo_data_f.append(Hb_f[0, :])
            hemo_data_f.append(Hb_f[1, :])
            hemo_data_f.append(Hb_f[0, :] + Hb_f[1, :])

            #Channel Names
            hemo_chan_red = pair + ' HbO'
            hemo_chan_ir = pair + ' HbR'
            hemo_chan_total = pair + ' HbT'
            hemo_chanLIST.append(hemo_chan_red)
            hemo_chanLIST.append(hemo_chan_ir)
            hemo_chanLIST.append(hemo_chan_total)

        self.hemoDF = pd.DataFrame(np.column_stack(hemo_data), columns=hemo_chanLIST)
        self.hemoDF_f = pd.DataFrame(np.column_stack(hemo_data_f), columns=hemo_chanLIST)

    def bandpass_filter(self, data_type, filter_cutoffs=(0.01, 0.2), transition_width=0.01, numtaps=30001,freq_plot=False):

        dataDF, chan_list = self.get_data(data_type)

        """Setup Filter"""
        fs = self.fs
        filt_data = []
        filter_weights = signal.firwin(numtaps, filter_cutoffs, width=transition_width, window='Hamming',
                                       pass_zero='bandpass', fs=fs)
        if freq_plot:
            w, h = signal.freqz(filter_weights, worN=fft.next_fast_len(40000, real=True))
            plt.plot((w / np.pi) * (fs / 2), 20 * np.log10(np.abs(h)))
            plt.xlim((0, 2))
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude (dB)')
            plt.show()

        for chan in chan_list:
            data = dataDF[chan].to_numpy()
            padded_data = pad_noise(data, numtaps, 5000)
            filtered_data = np.flip(
                signal.fftconvolve(np.flip(signal.fftconvolve(padded_data, filter_weights, mode='same')),
                                   filter_weights, mode='same'))
            filt_data.append(filtered_data[numtaps:-numtaps])

        # self.raw_data_f = pd.DataFrame(np.column_stack(filt_data), columns = chan_list)
        self.overwrite_data(data_to_write = pd.DataFrame(filt_data, columns=chan_list), data_type=data_type)

    def calc_hr(self, channel, filter_cutoffs = (1,40), peak_height = None, peak_distance = None, transition_width=0.01, numtaps=3001, smoothing_window = 3, check_plot=False):
        fs = self.fs
        filter_weights = signal.firwin(numtaps, filter_cutoffs, width=transition_width, window='Hamming',
                                       pass_zero='bandpass', fs=fs)

        w, h = signal.freqz(filter_weights, worN=fft.next_fast_len(40000, real=True))

        data = self.raw_data[channel].to_numpy()
        padded_data = pad_noise(data, numtaps, 5000)
        data_f = np.flip(
            signal.fftconvolve(np.flip(signal.fftconvolve(padded_data, filter_weights, mode='same')), filter_weights,
                               mode='same'))[numtaps:-numtaps]

        if peak_height is not None:
            height = peak_height
        else:
            height = 1000

        if peak_distance is not None:
            distance = peak_distance
        else:
            distance = fs * 0.25

        peaks, _ = signal.find_peaks(data_f, height = height, distance = distance)
        peaks = peaks[1:-1] #drops first/last values of peaks due to risk of odd behaviors at start/end of recording

        if check_plot:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=self.time, y=data_f))
            fig.add_trace(go.Scatter(x=self.time[peaks], y=data_f[peaks], mode='markers'))
            fig.show()

        peak_dt = np.diff(peaks) / self.fs  # Time between peaks in seconds

        idx = [int((peaks[i] + peaks[i + 1]) / 2) for i in np.arange(len(peaks) - 1)]
        peak_time = self.time[idx]
        bpm = 60 / peak_dt  # Instantaneous Heart rate in BPM based on peak_dt

        # Construct ECG dataframe?
        ecg_dct = {'peak_idx': idx,
                   'peak_dt': peak_dt,
                   'Time (s)': peak_time,
                   'b2b bpm': bpm}

        self.ecgDF = pd.DataFrame(ecg_dct)
        self.ecgDF['avg. bpm'] = self.ecgDF['b2b bpm'].rolling(window=smoothing_window).mean()

    def ssr_regression(self, data_type):

        dataDF, channels = self.get_data(data_type)
        red_channels = ['D1 LS Red', 'D1 LL Red', 'D3 LS Red', 'D3 LL Red']
        ir_channels = ['D1 LS IR', 'D1 LL IR', 'D3 LS IR', 'D3 LL IR']
        ss_red_ch = 'SS Red'
        ss_ir_ch = 'SS IR'

        if 'Hemo' in data_type:
            red_channels = ['D1 LS HbO', 'D1 LL HbO', 'D3 LS HbO', 'D3 LL HbO']
            ir_channels = ['D1 LS HbR', 'D1 LL HbR', 'D3 LS HbR', 'D3 LL HbR']
            HbT_channels = ['D1 LS HbT', 'D1 LL HbT', 'D3 LS HbT', 'D3 LL HbT']
            ss_red_ch = 'SS HbO'
            ss_ir_ch = 'SS HbR'
            ss_HbT_ch = 'SS HbT'

        ssr_data = []
        ssr_ch_name = []
        
        #Cleaning Red Channels
        ss_red = dataDF[ss_red_ch].to_numpy()
        for chan in red_channels:
            d = dataDF[chan].to_numpy()
            alpha = np.dot(ss_red, d) / np.dot(ss_red,ss_red)
            d_ssr = d - (alpha * ss_red)
            ssr_ch_name.append(chan + ' SSR')
            ssr_data.append(d_ssr)

        # Cleaning IR Channels
        ss_ir = dataDF[ss_ir_ch].to_numpy()
        for chan in ir_channels:
            d = dataDF[chan].to_numpy()
            alpha = np.dot(ss_ir, d) / np.dot(ss_ir, ss_ir)
            d_ssr = d - (alpha * ss_ir)
            ssr_ch_name.append(chan + ' SSR')
            ssr_data.append(d_ssr)

        if 'Hemo' in data_type:
            ss_HbT = dataDF[ss_HbT_ch].to_numpy()
            for chan in HbT_channels:
                d = dataDF[chan].to_numpy()
                alpha = np.dot(ss_HbT, d) / np.dot(ss_HbT, ss_HbT)
                d_ssr = d - (alpha * ss_HbT)
                ssr_ch_name.append(chan + ' SSR')
                ssr_data.append(d_ssr)

        if 'filt' in data_type:
            self.ssrDF_f = pd.DataFrame(np.column_stack(ssr_data), columns = ssr_ch_name)
        else:
            self.ssrDF = pd.DataFrame(np.column_stack(ssr_data), columns=ssr_ch_name)
            
    def get_data(self, data_type = None):
        channels = ['SS Red', 'SS IR', 'D1 LS Red', 'D1 LL Red', 'D1 LS IR', 'D1 LL IR', 
                    'D3 LS Red', 'D3 LL Red', 'D3 LS IR', 'D3 LL IR']
        if data_type == 'raw':
            dataDF = self.raw_data.copy()
        elif data_type == 'raw_filt':
            dataDF = self.raw_data_f.copy()
        elif data_type == 'OD':
            dataDF = self.d_OD.copy()
        elif data_type == 'OD_filt':
            dataDF = self.d_OD_filt.copy()
        elif data_type == 'Mua':
            dataDF = self.d_Mua.copy()
        elif data_type == 'Mua_filt':
            dataDF = self.d_Mua_filt.copy()
        elif data_type == 'Hemo':
            channels = ['SS HbO', 'SS HbR', 'D1 LS HbO', 'D1 LL HbO', 'D1 LS HbR', 'D1 LL HbR',
                        'D3 LS HbO', 'D3 LL HbO', 'D3 LS HbR', 'D3 LL HbR']
            dataDF = self.hemoDF.copy()
        elif data_type == 'Hemo_filt':
            channels = ['SS HbO', 'SS HbR', 'D1 LS HbO', 'D1 LL HbO', 'D1 LS HbR', 'D1 LL HbR',
                        'D3 LS HbO', 'D3 LL HbO', 'D3 LS HbR', 'D3 LL HbR']
            dataDF = self.hemoDF_f.copy()
        elif data_type == 'SSR':
            dataDF = self.ssrDF.copy()
        elif data_type == 'SSR_filt':
            dataDF =self.ssrDF_f.copy()
        else:
            raise ValueError("Data type not recognized. Data type must be 'raw', 'OD', 'Mua', 'Hemo', or 'SSR'. With '_filt' added for filtered versions.")

        return dataDF, channels

    def overwrite_data(self, data_to_write, data_type = None):

        if data_type == 'raw':
            self.raw_data = data_to_write
        elif data_type == 'raw_filt':
            self.raw_data_f  = data_to_write
        elif data_type == 'OD':
            self.d_OD  = data_to_write
        elif data_type == 'OD_filt':
            self.d_OD_filt  = data_to_write
        elif data_type == 'Mua':
            self.d_Mua  = data_to_write
        elif data_type == 'Mua_filt':
            self.d_Mua_filt = data_to_write
        elif data_type == 'Hemo':
            self.hemoDF  = data_to_write
        elif data_type == 'Hemo_filt':
            self.hemoDF_f  = data_to_write
        elif data_type == 'SSR':
            self.ssrDF = data_to_write
        elif data_type == 'SSR_filt':
            self.ssrDF_f = data_to_write
        else:
            raise ValueError(
                "Data type not recognized. Data type must be 'raw', 'OD', 'Mua','Hemo', or 'SSR', with '_filt' added for filtered versions.")

    def plot_artifact(self, channel, fc = 1, width = 1, numtaps = 3001, filter_type = 'highpass', show_stim = False):

        filter_cutoff = fc
        transition_width = width

        filter_weights = signal.firwin(numtaps, filter_cutoff, width=transition_width, window='Hamming',
                                       pass_zero= filter_type, fs=self.fs)
        w, h = signal.freqz(filter_weights, worN=fft.next_fast_len(40000, real=True))

        """flexNIRs High-pass Artifact Filtering -- WIP"""
        data_cols = ['D1 Ambient', 'D3 Ambient']
        artDF =pd.DataFrame()
        artDF['Time (s)'] = self.time

        for col in data_cols:
            data = self.ambient_data[col].to_numpy()
            padded_data = pad_noise(data, numtaps, 5000)
            filtered_data = np.flip(
                signal.fftconvolve(np.flip(signal.fftconvolve(padded_data, filter_weights, mode='same')), filter_weights,
                                   mode='same'))
            name = col
            artDF[name] = filtered_data[numtaps:-numtaps]

        """Plotly plot for looking at stim artifact in flexNIRs data"""
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=artDF['Time (s)'], y=artDF[channel], customdata=artDF.index, hovertemplate='%{customdata:.1f}'))

        if show_stim == True:
            if self.stimDF is not None:
                for param in self.stimDF.index:
                    fig.add_vrect(x0=self.stimDF.loc[param]['fNIRs onset time (s)'],
                                  x1=self.stimDF.loc[param]['fNIRs offset time (s)'],
                                  line_width=0, fillcolor='red', opacity=0.15)
            else:
                raise ValueError('Stimulation alignment data not found.')
        fig.show()

    def plot_channel(self, data_type, channel, plot_style = 'stacked', pre_time=5, post_time=30, zero_shift = False, fig_size = (10,10),
                     show = True, legend = False, title = None):

        fs = self.fs
        plotDF = self.get_data(data_type)[0]

        #channel_pair = self.ch_pairs[channel]

        if 'SSR' in data_type:
            channel_pair = [ self.ch_dct[ch][0] + ' ' + self.ch_dct[ch][1] + ' SSR' for ch in self.ch_pairs[channel]]
            channel_pair = [ch.replace('Red', 'HbO') for ch in channel_pair]
            channel_pair = [ch.replace('IR', 'HbR') for ch in channel_pair]
        elif 'Hemo' in data_type:
            channel_pair = [ self.ch_dct[ch][0] + ' ' + self.ch_dct[ch][2] for ch in self.ch_pairs[channel]]
        else:
            channel_pair = [self.ch_dct[ch][0] + ' ' + self.ch_dct[ch][1] for ch in self.ch_pairs[channel]]

        pre_BL_idx_width = int(pre_time * fs)
        post_BL_idx_width = int(post_time * fs)

        if self.stimDF is None:
            raise ValueError('Stimulation alignment data not found.')
        else:
            stimDF = self.stimDF.copy()
            # Label data points based on stimulation #
            for idx, stim in enumerate(stimDF.index):
                stim_start_idx = stimDF.loc[stim, 'fNIRs onset index']
                stim_stop_idx = stimDF.loc[stim, 'fNIRs offset index']

                plot_start_idx = (stim_start_idx - pre_BL_idx_width).astype(int)
                plot_stop_idx = (stim_stop_idx + post_BL_idx_width).astype(int)

                if plot_stop_idx > len(plotDF):
                    raise Exception('Plot stop index exceeds plot length. Reduce post-stimulation time.')

                plotDF.loc[plot_start_idx:plot_stop_idx, 'Stim #'] = str(idx + 1)

                # Construct time array based on sampling frequency and number of indices with time zero at stimulation start
                time_span = plot_stop_idx - plot_start_idx + 1
                time = np.arange(-pre_BL_idx_width, time_span - pre_BL_idx_width, 1) / fs

                plotDF.loc[plot_start_idx:plot_stop_idx, 'Trial Time'] = time

            plotDF.dropna(axis=0, subset=['Stim #'], inplace=True)

            if plot_style == 'stacked':

                fig, ax = plt.subplots(figsize=fig_size, nrows=2)
                ax = ax.ravel()
                for idx, chan in enumerate(channel_pair):

                    # Shifts data so all trials start at 0
                    if zero_shift:
                        for stim_num in plotDF['Stim #'].unique():
                            zero_point = plotDF.loc[plotDF['Stim #'] == stim_num, 'Trial Time'].abs().idxmin()
                            plotDF.loc[plotDF['Stim #'] == stim_num, chan] = plotDF.loc[
                                                                                 plotDF['Stim #'] == stim_num, chan] - \
                                                                             plotDF.loc[zero_point, chan].item()

                    sns.lineplot(data=plotDF, x='Trial Time', y=chan, hue='Stim #', ax=ax[idx], legend=legend)
                    ax[idx].set_title(chan)
                    ax[idx].set_xlabel('Time (s)')
                    ax[idx].set_ylabel('A.U.')

                # This shading assumes that all stimulations within file were the same duration.  It uses the duration of the
                # last stim only
                ax[0].axvspan(xmin=0, xmax=stimDF.loc[stim, 'duration (ms)'] * 1e-3, color='gray', alpha=0.2)
                ax[1].axvspan(xmin=0, xmax=stimDF.loc[stim, 'duration (ms)'] * 1e-3, color='gray', alpha=0.2)

            elif plot_style== 'average':
                fig, ax = plt.subplots(figsize=fig_size)

                for idx, chan in enumerate(channel_pair):
                    # Shifts data so all trials start at 0
                    if zero_shift:
                        for stim_num in plotDF['Stim #'].unique():
                            zero_point = plotDF.loc[plotDF['Stim #'] == stim_num, 'Trial Time'].abs().idxmin()
                            plotDF.loc[plotDF['Stim #'] == stim_num, chan] = (
                                    plotDF.loc[plotDF['Stim #'] == stim_num, chan] - plotDF.loc[zero_point, chan].item())

                    sns.lineplot(data=plotDF, x='Trial Time', y=chan, errorbar='sd', ax=ax, label=chan)
                ax.axvspan(xmin=0, xmax=stimDF.loc[stim, 'duration (ms)'] * 1e-3, color='gray', alpha=0.2)

            elif plot_style == 'Full':

                fig, ax = plt.subplots(figsize=fig_size, nrows=3, sharex=True)
                ax = ax.ravel()
                for idx, chan in enumerate(channel_pair):

                    if zero_shift:
                        for stim_num in plotDF['Stim #'].unique():
                            zero_point = plotDF.loc[plotDF['Stim #'] == stim_num, 'Trial Time'].abs().idxmin()
                            plotDF.loc[plotDF['Stim #'] == stim_num, chan] = (
                                    plotDF.loc[plotDF['Stim #'] == stim_num, chan] - plotDF.loc[zero_point, chan].item())

                    sns.lineplot(data=plotDF, x='Trial Time', y=chan, errorbar='sd', ax=ax[0], label=chan)
                ax[0].axvspan(xmin=0, xmax=stimDF.loc[stim, 'duration (ms)'] * 1e-3, color='gray', alpha=0.2)
                ax[0].set_title(channel + ' Trial Average (mean ' + r'$\pm$' + ' s.d.)')
                ax[0].set_ylabel('A.U.')
                ax[0].legend(loc='upper right')

                for idx, chan in enumerate(channel_pair):

                    # Shifts data so all trials start at 0
                    if zero_shift:
                        for stim_num in plotDF['Stim #'].unique():
                            zero_point = plotDF.loc[plotDF['Stim #'] == stim_num, 'Trial Time'].abs().idxmin()
                            plotDF.loc[plotDF['Stim #'] == stim_num, chan] = plotDF.loc[
                                                                                 plotDF['Stim #'] == stim_num, chan] - \
                                                                             plotDF.loc[zero_point, chan].item()

                    sns.lineplot(data=plotDF, x='Trial Time', y=chan, hue='Stim #', ax=ax[idx + 1], legend=legend)
                    ax[idx + 1].set_title(chan)
                    ax[idx + 1].set_xlabel('Time (s)')
                    ax[idx + 1].set_ylabel('A.U.')

                # This shading assumes that all stimulations within file were the same duration.  It uses the duration of the
                # last stim only
                ax[1].axvspan(xmin=0, xmax=stimDF.loc[stim, 'duration (ms)'] * 1e-3, color='gray', alpha=0.2)
                ax[2].axvspan(xmin=0, xmax=stimDF.loc[stim, 'duration (ms)'] * 1e-3, color='gray', alpha=0.2)
                if legend:
                    ax[1].legend(loc='upper right')
                    ax[2].legend(loc='upper right')

            fig.tight_layout()

            if title:
                fig.suptitle(title)

            if show:
                backend = mpl_get_backend()
                fig.tight_layout()
                if backend == "module://ipympl.backend_nbagg":
                    fig.canvas.header_visible = False
                    fig.canvas.footer_visible = False
                    fig.show()
                else:
                    plt.show()
                    plt.close(fig)
                return None
            else:
                return ax

    def ssr_plot(self, data_type, channel, show_stim = False, show_ss = True, show_hr = False, hr_chan = 'b2b'):

        ssr_channel = channel + ' SSR'

        d = self.get_data(data_type)[0]

#        ch_name = self.name_dct[channel]

        #if the data in ssrDF is hemoglobin, needs to change the name to include HbO instead of Red or HbR instead of IR
        #need to check if 'Hb' is in the column names
        if 'Hb' in d.columns[0]:
            if 'Red' in channel:
                channel = channel.replace('Red', 'HbO')
                ssr_channel = ssr_channel.replace('Red', 'HbO')
            elif 'IR' in channel:
                channel = channel.replace('IR', 'HbR')
                ssr_channel = ssr_channel.replace('Red', 'HbO')

        if 'filt' in data_type:
            ssr_d = self.get_data(data_type = 'SSR_filt')[0]
        else:
            ssr_d = self.get_data(data_type = 'SSR')[0]

        if 'Red' in channel:

            if 'Hemo' in data_type:
                ss_channel = 'SS HbO'
            else:
                ss_channel = 'SS Red'

        elif 'IR' in channel:

            if 'Hemo' in data_type:
                ss_channel = 'SS HbR'
            else:
                ss_channel = 'SS IR'
        elif 'HbT' in channel:
            ss_channel = 'SS HbT'

        print('Data Channel: ' + channel)
        print('Short-Separation Channel: ' + ss_channel)

        # Construct dictionary of original data, SSR regressed data, and SS channel for plotting
        if show_ss:
            plot_dct = {'Orig. Data' : d[channel], #Non-regressed data
                        'SSR Data' : ssr_d[ssr_channel], #Data with short-channel regression
                        'SS Data' : d[ss_channel], #Short-channel from non-regressed data
                        }
        else:
            plot_dct = {'Orig. Data' : d[channel], #Non-regressed data
                        'SSR Data' : ssr_d[ssr_channel], #Data with short-channel regression
                        }

        #fig = go.Figure()
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        for data in plot_dct:
            fig.add_trace(go.Scatter(x = self.time, y = plot_dct[data], name = data))

        if show_hr:

            if hr_chan == 'b2b':
                hr_plot_chan = 'b2b bpm'
            elif hr_chan == 'smooth':
                hr_plot_chan = 'avg. bpm'
            fig.add_trace(go.Scatter(x = self.ecgDF['Time (s)'], y = self.ecgDF[hr_plot_chan], name = 'HR (bpm)', line_color = 'gray', opacity = 0.75), secondary_y=True)

        if show_stim:
            if self.stimDF is not None:
                for param in self.stimDF.index:
                    fig.add_vrect(x0=self.stimDF.loc[param]['fNIRs onset time (s)'],
                                  x1=self.stimDF.loc[param]['fNIRs offset time (s)'],
                                  line_width=0, fillcolor='red', opacity=0.15)
            else:
                raise ValueError('Stimulation alignment data not found.')
        fig.show()

    def plot_channel_interactive(self, data_type, channel, show_stim = False, show_hr = False, hr_chan = 'b2b'):

        d = self.get_data(data_type)[0][channel]

        dct = {'Data':d, 'Time (s)':self.time}

        plotDF = pd.DataFrame(data=dct)

        #fig = go.Figure()
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(x=plotDF['Time (s)'], y=plotDF['Data'], customdata=plotDF.index,
                       hovertemplate='%{customdata:.1f}', name = channel))

        if show_hr:

            if hr_chan == 'b2b':
                hr_plot_chan = 'b2b bpm'
            elif hr_chan == 'smooth':
                hr_plot_chan = 'avg. bpm'
            fig.add_trace(go.Scatter(x = self.ecgDF['Time (s)'], y = self.ecgDF[hr_plot_chan], name = 'HR (bpm)', line_color = 'gray', opacity = 0.75), secondary_y=True)

        if show_stim == True:
            if self.stimDF is not None:
                for param in self.stimDF.index:
                    fig.add_vrect(x0=self.stimDF.loc[param]['fNIRs onset time (s)'],
                                  x1=self.stimDF.loc[param]['fNIRs offset time (s)'],
                                  line_width = 0, fillcolor='red', opacity = 0.15)
            else:
                raise ValueError('Stimulation alignment data not found.')
        fig.show()

    def gigaplot(self, data_type, channel, pre_time = 0, post_time = 0, zero_shift = False, show_stim = False, show_hr = False, hr_chan = 'b2b', fig_size=(10,5)):
        fs = self.fs
        plotDF = self.get_data(data_type)[0]
        plotDF['Time (s)'] = self.time

        if 'SSR' in data_type:
            channel_pair = [self.ch_dct[ch][0] + ' ' + self.ch_dct[ch][1] + ' SSR' for ch in self.ch_pairs[channel]]
            channel_pair = [ch.replace('Red', 'HbO') for ch in channel_pair]
            channel_pair = [ch.replace('IR', 'HbR') for ch in channel_pair]
        elif 'Hemo' in data_type:
            channel_pair = [self.ch_dct[ch][0] + ' ' + self.ch_dct[ch][2] for ch in self.ch_pairs[channel]]
        else:
            channel_pair = [self.ch_dct[ch][0] + ' ' + self.ch_dct[ch][1] for ch in self.ch_pairs[channel]]

        pre_BL_idx_width = int(pre_time * fs)
        post_BL_idx_width = int(post_time * fs)

        if self.stimDF is None:
            raise ValueError('Stimulation alignment data not found.')
        else:
            stimDF = self.stimDF.copy()
            # Label data points based on stimulation #
            for idx, stim in enumerate(stimDF.index):
                stim_start_idx = stimDF.loc[stim, 'fNIRs onset index']
                stim_stop_idx = stimDF.loc[stim, 'fNIRs offset index']

                plot_start_idx = (stim_start_idx - pre_BL_idx_width).astype(int)
                plot_stop_idx = (stim_stop_idx + post_BL_idx_width).astype(int)

                if plot_stop_idx > len(plotDF):
                    raise Exception('Plot stop index exceeds plot length. Reduce post-stimulation time.')

                plotDF.loc[plot_start_idx:plot_stop_idx, 'Stim #'] = str(idx + 1)

                # Construct time array based on sampling frequency and number of indices with time zero at stimulation start
                time_span = plot_stop_idx - plot_start_idx + 1
                time = np.arange(-pre_BL_idx_width, time_span - pre_BL_idx_width, 1) / fs

                plotDF.loc[plot_start_idx:plot_stop_idx, 'Trial Time'] = time

        #Construct super figure
        fig = plt.figure(layout='constrained', figsize=fig_size)

        # mosaic = [['full', 'full', 'full', 'full'],
        #           ['D1 LS avg', 'D1 LL avg', 'D3 LS avg', 'D3 LL avg'],
        #           ['D1 LS Red', 'D1 LL Red', 'D3 LS Red', 'D3 LL Red'],
        #           ['D1 LS IR', 'D1 LL IR', 'D3 LS IR', 'D3 LL IR'], ]

        mosaic = [['full','full','full'],
                  ['avg', 'Red', 'IR']]

        ax_dict = fig.subplot_mosaic(mosaic)

        #Top Plot = Whole trace with shaded stim boxed
        for chan in channel_pair:
            if 'HbO' in chan:
                label_txt = 'HbO'
            elif 'HbR' in chan:
                label_txt = 'HbR'
            elif 'IR' in chan:
                label_txt = 'IR'
            elif 'Red' in chan:
                label_txt = 'Red'

            # Top Plot = Whole trace with shaded stim boxed
            sns.lineplot(data = plotDF, x = plotDF['Time (s)'], y = plotDF[chan], ax = ax_dict['full'], label = label_txt)

            if show_stim:
                for param in stimDF.index:
                    ax_dict['full'].axvspan(xmin=stimDF.loc[param]['fNIRs onset time (s)'],
                                            xmax=stimDF.loc[param]['fNIRs offset time (s)'], color='red', alpha=0.05)

        plotDF.dropna(axis=0, subset=['Stim #'], inplace=True)
        for chan in channel_pair:

            if 'HbO' in chan:
                label_txt = 'HbO'
            elif 'HbR' in chan:
                label_txt = 'HbR'
            elif 'IR' in chan:
                label_txt = 'IR'
            elif 'Red' in chan:
                label_txt = 'Red'

        # Avg. Response during stim Plot
            if zero_shift:
                for stim_num in plotDF['Stim #'].unique():
                    zero_point = plotDF.loc[plotDF['Stim #'] == stim_num, 'Trial Time'].abs().idxmin()
                    plotDF.loc[plotDF['Stim #'] == stim_num, chan] = (
                            plotDF.loc[plotDF['Stim #'] == stim_num, chan] - plotDF.loc[
                        zero_point, chan].item())

            sns.lineplot(data=plotDF, x='Trial Time', y=chan, errorbar='sd', ax=ax_dict['avg'], label=label_txt)

            #Individual stacked trace plots
            if ('Red' in chan) or ('HbO' in chan):
                sns.lineplot(data=plotDF, x='Trial Time', y=chan, hue='Stim #', ax=ax_dict['Red'], legend = False)
            elif ('IR' in chan) or ('HbR' in chan):
                sns.lineplot(data=plotDF, x='Trial Time', y=chan, hue='Stim #', ax=ax_dict['IR'], legend = False)

            ax_dict['avg'].axvspan(xmin=0, xmax=stimDF.loc[stim, 'duration (ms)'] * 1e-3, color='red', alpha=0.05)
            ax_dict['Red'].axvspan(xmin=0, xmax=stimDF.loc[stim, 'duration (ms)'] * 1e-3, color='red', alpha=0.05)
            ax_dict['IR'].axvspan(xmin=0, xmax=stimDF.loc[stim, 'duration (ms)'] * 1e-3, color='red', alpha=0.05)

    def _ch_to_index(self, ch):
        """
        Returns a boolean array with indices corresponding to the self.ch_names property. A value in the array will be
        'True' if the channel name is present in the 'ch' argument, otherwise the value will be 'False'.

        Parameters
        ----------
        ch : str, list
            Channel name or names for the method to search for.

        Returns
        -------
        numpy.ndarray
            A numpy array of boolean values.
        See Also
        --------
        ch_names
        """
        # Use numpy to check of ch is string or list of strings
        ch = np.asarray(ch).flatten()
        if ch.dtype.type is np.bool_:
            if len(ch) == len(self.ch_names):
                return ch
            else:
                raise ValueError(
                    "Length of boolean np.ndarray does not match length of channels"
                )
        else:
            if ch.dtype.type is np.str_:
                ch_compare = np.asarray(self.ch_names)
            else:
                ch = _to_numeric_array(ch)
                ch_compare = np.arange(0, len(self.ch_names))
            extra_ch = np.isin(ch, ch_compare, invert=True)
            if np.any(extra_ch):
                warnings.warn(
                    str(ch[extra_ch])
                    + " not found in channel list or is outside range."
                )
            return np.isin(ch_compare, ch)

def _to_numeric_array(array, dtype=float):
    """
    Convert python objects to a 1D numeric array.

    Converts a python object into a numeric numpy array. Utilizes numpy's np.asarray and np.astype in order to
    gracefully handle different object types as well as raise appropriate error messages. Always flattens result to
    a 1D array.

    Parameters
    ----------
    array : object
        Input object which will be converted to a numpy array.
    dtype : str, type
        The dtype of the array that will be returned.

    Returns
    -------
    np.ndarray
        A 1D numeric array with type given by 'dtype'.

    Raises
    ------
    ValueError
        If the conversion to a numpy array or the dtype conversion fails.
    """
    try:
        np_array = np.asarray(array)

        # Convert type if necessary
        if np_array.dtype != dtype:
            np_array = np_array.astype(dtype)

        # Flatten the array if it's not already 1D
        if np_array.ndim != 1:
            np_array = np_array.flatten()

        return np_array
    except Exception as e:
        raise ValueError(f"Conversion to numeric array failed: {e}")

def _plt_setup_fig_axis(axis=None, fig_size=(5, 3), subplots=(1, 1), **kwargs):
    """Convenience function to setup a figure axis

    Parameters
    ----------
    axis : None, matplotlib.axis.Axis
        Either None to use a new axis or matplotlib axis to plot on.
    fig_size : tuple, list, np.ndarray
        The size (width, height) of the matplotlib figure.
    subplots : tuple
        The num rows, num columns of axis to plot on.
    **kwargs
        Keyword arguments to pass to fig.add_subplot.
    Returns
    -------
    fig : matplotlib.figure.Figure
        matplotlib figure reference.
    ax : matplotlib.axis.Axis
        matplotlib axis reference.
    """
    plt.ioff()
    if axis is None:
        fig, ax = plt.subplots(*subplots, figsize=fig_size, **kwargs)
    else:
        fig = axis.figure
        ax = axis
    return fig, ax

def _plt_show_fig(fig, ax, show):
    """Convenience function to show a figure axis.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        matplotlib figure reference.
    ax : matplotlib.axis.Axis
        matplotlib axis reference.
    show: bool
        Boolean value indicating if the plot should be displayed.
    Returns
    -------
    ax : matplotlib.axis.Axis
        matplotlib axis reference.
    """
    if show:
        backend = mpl_get_backend()
        fig.tight_layout()
        if backend == "module://ipympl.backend_nbagg":
            fig.canvas.header_visible = False
            fig.canvas.footer_visible = False
            fig.show()
        else:
            plt.show()
            plt.close(fig)
        return None
    else:
        return ax

def pad_noise(array, pad_length, pad_calc_length):
    head_loc = np.mean(array[0:pad_calc_length])
    tail_loc = np.mean(array[-pad_calc_length])
    noise_scale = np.std(array)

    head_noise_array = np.random.normal(head_loc, noise_scale, pad_length)
    tail_noise_array = np.random.normal(tail_loc, noise_scale, pad_length)

    new_array = np.concatenate([head_noise_array, array, tail_noise_array])
    return new_array

def filter_hr(data, cutoff = 10, downsample = False, downsample_factor = 10, check_plot=False):

    raw_ecg = data.array.compute()[0] * 1e6
    data_f = data.filter_gaussian(Wn=cutoff, btype='lowpass')
    ecg_f = data_f.array.compute()[0] * 1e6

    time = data.time().compute()

    if downsample:
        ds_ECGr = signal.decimate(raw_ecg, downsample_factor, zero_phase = True)
        ds_time = signal.decimate(time, downsample_factor, zero_phase = True)
        ds_ECGf = signal.decimate(ecg_f, downsample_factor, zero_phase = True)

        """Check alignment of filtered and downsampled signal against original"""
        if check_plot:
            fig = make_subplots(specs = [[{'secondary_y' : True}]])
            fig.add_trace(go.Scatter(x = ds_time,y = ds_ECGr, name = 'raw'))
            fig.add_trace(go.Scatter(x = ds_time,y = ds_ECGf, name = 'Filtered'), secondary_y = True)
            fig.show()

        d = ds_ECGf
        t = ds_time

    else:
        d = ecg_f
        t = time

        """Check alignment of filtered and downsampled signal against original"""
        if check_plot:
            fig = make_subplots(specs = [[{'secondary_y' : True}]])
            fig.add_trace(go.Scatter(x = time,y = raw_ecg, name = 'raw'))
            fig.add_trace(go.Scatter(x = time,y = ecg_f, name = 'Filtered'), secondary_y = True)
            fig.show()

    data_array = np.vstack((d,t))
    return data_array

def calc_hr(data, fs, peak_height, smoothing_window_length = 5, check_plot = False):
    """Calculate ECG"""
    d = data[0]
    time = data[1]
    peaks = signal.find_peaks(d, height = peak_height, distance = fs * 0.25)[0]
    peaks = peaks[1:-1]

    """Get distance (time) between peaks and calculate heart rate"""
    peak_dt = np.diff(peaks) / fs #Time between peaks in seconds

    idx = [ int((peaks[i] + peaks[i + 1]) / 2) for i in np.arange(len(peaks) - 1)]
    peak_time = time[idx]
    bpm = 60 / peak_dt #Instantaneous Heart rate in BPM based on peak_dt

    #Construct ECG dataframe?
    ecg_dct = {'peak_idx' : idx,
               'peak_dt' : peak_dt,
               'Time (s)' : peak_time,
               'b2b bpm' : bpm}

    ecgDF = pd.DataFrame(ecg_dct)
    ecgDF['smooth bpm'] = ecgDF['b2b bpm'].rolling(smoothing_window_length).mean()

    if check_plot:
        #fig = go.Figure()
        fig = make_subplots(specs = [[{"secondary_y" : True}]])
        fig.add_trace(go.Scatter(x = time[peaks], y = d[peaks], mode = 'markers'), secondary_y = True)
        fig.add_trace(go.Scatter(x = time,y = d, name = 'ECG Trace'), secondary_y = True)
        fig.add_trace(go.Scatter(x = ecgDF['Time (s)'], y = ecgDF['b2b bpm'], name = 'HR (bpm)'))#, secondary_y=True)
        fig.show()

    return ecgDF