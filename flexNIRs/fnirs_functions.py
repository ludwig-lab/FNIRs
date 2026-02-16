import h5py
import numpy as np
import pandas as pd
import scipy.io as sio
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib import get_backend as mpl_get_backend
from scipy import signal
from scipy import fft
import plotly.graph_objs as go
from plotly.subplots import make_subplots

class flexNIRs():

    def __init__(self, filePATH):
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

        self.channels = ['SS Red', 'SS IR',
                         'D1 LS Red', 'D1 LL Red', 'D1 LS IR', 'D1 LL IR',
                         'D3 LS Red', 'D3 LL Red', 'D3 LS IR', 'D3 LL IR']

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
        self.d_OD = pd.DataFrame(flex_data['deltaOD_raw'], columns = self.channels)
        self.d_Mua = pd.DataFrame(flex_data['deltaMua_raw'], columns = self.channels)

        #Filtered OD & Mua calculated by flexNIRs MATLAB Script
        self.d_OD_filt = pd.DataFrame(flex_data['deltaOD_filt'], columns = self.channels)
        self.d_Mua_filt = pd.DataFrame(flex_data['deltaMua_filt'], columns = self.channels)

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

            #Filtered Mua Values
            data_red_f = self.d_Mua_filt[ch_red].values
            data_ir_f = self.d_Mua_filt[ch_ir].values
            d_f = np.vstack((data_red_f, data_ir_f))
            Hb_f = np.matmul(exC ** -1, d_f)

            hemo_data_f.append(Hb_f[0, :])
            hemo_data_f.append(Hb_f[1, :])

            #Channel Names
            hemo_chan_red = pair + ' HbO'
            hemo_chan_ir = pair + ' HbR'
            hemo_chanLIST.append(hemo_chan_red)
            hemo_chanLIST.append(hemo_chan_ir)

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
            ss_red_ch = 'SS HbO'
            ss_ir_ch = 'SS HbR'

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

    def plot_artifact(self, channel, fc = 1, width = 1, numtaps = 3001, show_stim = False):
        filter_cutoff = fc
        transition_width = width

        filter_weights = signal.firwin(numtaps, filter_cutoff, width=transition_width, window='Hamming',
                                       pass_zero='highpass', fs=self.fs)
        w, h = signal.freqz(filter_weights, worN=fft.next_fast_len(40000, real=True))

        """flexNIRs High-pass Artifact Filtering -- WIP"""
        data_cols = ['D1 Ambient', 'D3 Ambient']
        artDF = self.ambient_data.copy()
        artDF['Time (s)'] = self.time

        for col in data_cols:
            data = artDF[col].to_numpy()
            padded_data = data  # pad_noise(data, numtaps, 5000)
            filtered_data = np.flip(
                signal.fftconvolve(np.flip(signal.fftconvolve(padded_data, filter_weights, mode='same')), filter_weights,
                                   mode='same'))
            name = col + ' Filtered'
            artDF[name] = filtered_data  # [numtaps:-numtaps]

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
        plotDF, _ = self.get_data(data_type)

        channel_pairs = {'SS': ['SS Red', 'SS IR'],
                         'D1 LS': ['D1 LS Red', 'D1 LS IR'],
                         'D1 LL': ['D1 LL Red', 'D1 LL IR'],
                         'D3 LS': ['D3 LS Red', 'D3 LS IR'],
                         'D3 LL': ['D3 LL Red', 'D3 LL IR'],}
        if 'SSR' in data_type:
            channel_pairs = {'SS': ['SS Red', 'SS IR'],
                             'D1 LS': ['D1 LS Red SSR', 'D1 LS IR SSR'],
                             'D1 LL': ['D1 LL Red SSR', 'D1 LL IR SSR'],
                             'D3 LS': ['D3 LS Red SSR', 'D3 LS IR SSR'],
                             'D3 LL': ['D3 LL Red SSR', 'D3 LL IR SSR'], }
        elif 'Hemo' in data_type:
            channel_pairs = {'SS': ['SS HbO', 'SS HbR'],
                             'D1 LS': ['D1 LS HbO', 'D1 LS HbR'],
                             'D1 LL': ['D1 LL HbO', 'D1 LL HbR'],
                             'D3 LS': ['D3 LS HbO', 'D3 LS HbR'],
                             'D3 LL': ['D3 LL HbO', 'D3 LL HbR'],}

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
                for idx, chan in enumerate(channel_pairs[channel]):

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
                for idx, chan in enumerate(channel_pairs[channel]):
                    sns.lineplot(data=plotDF, x='Trial Time', y=chan, errorbar='sd', ax=ax, label=chan)
                ax.axvspan(xmin=0, xmax=stimDF.loc[stim, 'duration (ms)'] * 1e-3, color='gray', alpha=0.2)

            elif plot_style == 'Full':

                fig, ax = plt.subplots(figsize=fig_size, nrows=3, sharex=True)
                ax = ax.ravel()
                for idx, chan in enumerate(channel_pairs[channel]):
                    sns.lineplot(data=plotDF, x='Trial Time', y=chan, errorbar='sd', ax=ax[0], label=chan)
                ax[0].axvspan(xmin=0, xmax=stimDF.loc[stim, 'duration (ms)'] * 1e-3, color='gray', alpha=0.2)
                ax[0].set_title(channel + ' Trial Average (mean ' + r'$\pm$' + ' s.d.)')
                ax[0].set_ylabel('A.U.')
                ax[0].legend(loc='upper right')

                for idx, chan in enumerate(channel_pairs[channel]):

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
        if 'filt' in data_type:
            ssr_d = self.get_data(data_type = 'SSR_filt')[0]
        else:
            ssr_d = self.get_data(data_type = 'SSR')[0]

        if 'Red' in channel:
            if 'Hemo' in data_type:
                ss_channel = 'SS HbO'
            else:
                ss_channel = 'SS Red'
        else:
            if 'Hemo' in data_type:
                ss_channel = 'SS HbR'
            else:
                ss_channel = 'SS IR'

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

def plux_import(filePATH):
    """"fNIRS Data Import -- for PLUX files"""
    fs = 1000
    data = {}
    h5_file = h5py.File(filePATH, 'r')

    for idx, hubID in enumerate(h5_file.keys()):
        for k in h5_file[hubID]['raw']:
            data['Hub_' + str(idx) + ' ' + k] = np.array(h5_file[hubID]['raw'][k]).flatten()
            # print(h5_file[hubID]['raw'][k])

            if h5_file[hubID]['raw'][k].name == '/' + hubID + '/raw/' + 'nSeq':
                data['Time (s)'] = np.array(h5_file[hubID]['raw'][k]).flatten() / fs
    fnirsDF = pd.DataFrame.from_dict(data)
    fnirsDF.rename(columns={'Hub_0 channel_10': 'IR', 'Hub_0 channel_9': 'Red', 'Hub_0 nSeq': 'Sample'}, inplace=True)
    fnirsDF = fnirsDF[['Sample', 'Time (s)', 'Red', 'IR']]
    return fnirsDF

def flex_import(filePATH):
    flex_data = sio.loadmat(filePATH)['data']
    flex_columns = ['Time (s)', 'SS Red', 'SS IR',
                    'D1 Ambient', 'D1 LS Red', 'D1 LL Red', 'D1 LS IR', 'D1 LL IR',
                    'D3 Ambient', 'D3 LS Red', 'D3 LL Red', 'D3 LS IR', 'D3 LL IR',
                    'HbT', 'SO2']

    flexDF = pd.DataFrame(flex_data, columns=flex_columns)
    return flexDF

def manual_alignment(flexDF, stimDF, stim_start_index):
    """Iterate through the stim dataframe and add stimulation start times to each row"""
    stim_index = stim_start_index
    stim_time = flexDF.loc[stim_index, 'Time (s)']

    """Get Indices of (approximate) times of all times and indices based on the first stim alignment"""
    fnirs_fs = 800 / 3
    fnirs_dt = 1 / fnirs_fs

    first_stim = stimDF.index[0][1]

    for stim_count, stim in enumerate(stimDF.index):

        # Sets up first stimulation data
        if stim[1] == first_stim:
            stimDF.loc[(0, first_stim), 'fNIRs onset index'] = stim_index
            stimDF.loc[(0, first_stim), 'fNIRs onset time (s)'] = stim_time

            stimDF.loc[(0, first_stim), 'fNIRs offset time (s)'] = stim_time + (
                        stimDF.loc[(0, first_stim), 'duration (ms)'] * 1e-3)

            stimDF.loc[(0, first_stim), 'fNIRs offset index'] = (stimDF.loc[(0, first_stim), 'fNIRs offset time (s)'] / fnirs_dt).astype(int)
            stim_idx_span = stimDF.loc[(0, first_stim), 'fNIRs offset index'] - stimDF.loc[(0, first_stim), 'fNIRs onset index']

        else:
            # Start of stimulation relative to first stimulation
            stim_start = stimDF.loc[stim, 'onset time (s)'] - stimDF.loc[(0, first_stim), 'onset time (s)']

            # Add time difference between current stim and first stim to fNIRs time
            stimDF.loc[stim, 'fNIRs onset time (s)'] = stimDF.loc[(0, first_stim), 'fNIRs onset time (s)'] + stim_start
            stimDF.loc[stim, 'fNIRs onset index'] = (stimDF.loc[stim, 'fNIRs onset time (s)'] / fnirs_dt).astype(int)

            # Add offset time based on duration of current stim
            stimDF.loc[stim, 'fNIRs offset time (s)'] = stimDF.loc[stim, 'fNIRs onset time (s)'] + (
                        stimDF.loc[stim, 'duration (ms)'] * 1e-3)

            """This ensures stimulation spans same # indices to account for discrepancies in sampling rates"""
            #stimDF.loc[stim, 'fNIRs offset index'] = (stimDF.loc[stim, 'fNIRs offset time (s)'] / fnirs_dt).astype(int)
            stimDF.loc[stim, 'fNIRs offset index'] = (stimDF.loc[stim, 'fNIRs onset index'] + stim_idx_span).astype(int)

    return stimDF


def align_flex_stim(flexDF, stimDF, threshold_multiplier = 2, plot = False):
    chanLIST = ['D1 Ambient', 'D3 Ambient']

    """Check which channel has the larger artifact by comparing variance"""
    var1 = flexDF[chanLIST[0]].var()
    var2 = flexDF[chanLIST[1]].var()

    if var1 > var2:
        d = flexDF[chanLIST[0]]
    else:
        d = flexDF[chanLIST[1]]

    """Rectify data in case first artifact is negative relative to baseline"""
    # Calculate raw mean of data
    d_mean = d.mean()
    # Shift all data to raw mean
    data = d - d_mean
    # Rectify data
    rData = data.abs()

    """Get first stim index based on a threshold value calculated from the data mean and standard deviation"""
    threshold = np.mean(rData) + (threshold_multiplier * np.std(rData))
    stim_index = np.min(np.nonzero(rData > threshold))
    stim_time = flexDF.loc[stim_index, 'Time (s)']

    """Get Indices of (approximate) times of all times and indices based on the first stim alignment"""
    fnirs_fs = 800 / 3
    fnirs_dt = 1 / fnirs_fs

    """Iterate through the stim dataframe and add stimulation start times to each row"""
    for stim_count, stim in enumerate(stimDF.index):

        # Sets up first stimulation data
        if stim_count == 0:
            stimDF.loc[(0, 0), 'fNIRs onset index'] = stim_index
            stimDF.loc[(0, 0), 'fNIRs onset time (s)'] = stim_time

            stimDF.loc[(0, 0), 'fNIRs offset time (s)'] = stim_time + (
                        stimDF.loc[(0, 0), 'duration (ms)'] * 1e-3)

            stimDF.loc[(0, 0), 'fNIRs offset index'] = (stimDF.loc[(0, 0), 'fNIRs offset time (s)'] / fnirs_dt).astype(int)
            stim_idx_span = stimDF.loc[(0, 0), 'fNIRs offset index'] - stimDF.loc[(0, 0), 'fNIRs onset index']

        else:
            # Start of stimulation relative to first stimulation
            stim_start = stimDF.loc[stim, 'onset time (s)'] - stimDF.loc[(0, 0), 'onset time (s)']

            # Add time difference between current stim and first stim to fNIRs time
            stimDF.loc[stim, 'fNIRs onset time (s)'] = stimDF.loc[(0, 0), 'fNIRs onset time (s)'] + stim_start
            stimDF.loc[stim, 'fNIRs onset index'] = (stimDF.loc[stim, 'fNIRs onset time (s)'] / fnirs_dt).astype(int)

            # Add offset time based on duration of current stim
            stimDF.loc[stim, 'fNIRs offset time (s)'] = stimDF.loc[stim, 'fNIRs onset time (s)'] + (
                        stimDF.loc[stim, 'duration (ms)'] * 1e-3)

            """This ensures stimulation spans same # indices to account for discrepancies in sampling rates"""
            #stimDF.loc[stim, 'fNIRs offset index'] = (stimDF.loc[stim, 'fNIRs offset time (s)'] / fnirs_dt).astype(int)
            stimDF.loc[stim, 'fNIRs offset index'] = (stimDF.loc[stim, 'fNIRs onset index'] + stim_idx_span).astype(int)

    if plot:
        fig,ax = plt.subplots(figsize=(12, 8))
        plt.plot(flexDF['Time (s)'], rData)
        ax.axhline(y = threshold, color='r')
        plt.show()

    return stimDF
    #return stim_vals

def align_plux_stim(dataDF, stimDF, threshold_multiplier = 3, plot=False):

    """High pass filter for finding artifact"""
    fs = 1000
    filter_cutoffs = [90, 110]
    transition_width = 1
    numtaps = 3001

    filter_weights = signal.firwin(numtaps, filter_cutoffs, width=transition_width, window='Hamming',
                                   pass_zero='bandpass', fs=fs)

    """PLUX High-pass Artifact Filtering -- WIP"""
    df = dataDF.copy()

    data_cols = ['Red', 'IR']  # For Single Hub
    for col in data_cols:
        data = df[col].to_numpy()
        padded_data = data
        filtered_data = np.flip(
            signal.fftconvolve(np.flip(signal.fftconvolve(padded_data, filter_weights, mode='same')), filter_weights,
                               mode='same'))
        name = col + ' Filtered'
        df[name] = filtered_data

    """Remove first/last 10s to remove for filter artifact"""
    df = df[10000:-10000]
    data = df['IR Filtered']

    data_mean = data.mean()
    data_std = data.std()
    threshold = data_mean + (data_std * threshold_multiplier)

    """Find first stim index and time"""
    stim_index = df.loc[df['IR Filtered'] > threshold].iloc[0]['Sample']
    stim_time = df.loc[stim_index, 'Time (s)']

    """Get remaining stim indices/times based on 1st index"""
    fnirs_dt = 1 / 1000

    for stim_count, stim in enumerate(stimDF.index):
        # Sets up first stimulation data
        if stim_count == 0:
            stimDF.loc[(0, 0), 'fNIRs onset time (s)'] = stim_time
            stimDF.loc[(0, 0), 'fNIRs onset index'] = stim_index

            stimDF.loc[(0, 0), 'fNIRs offset time (s)'] = stim_time + (
                    stimDF.loc[(0, 0), 'duration (ms)'] * 1e-3)

            stimDF.loc[(0, 0), 'fNIRs offset index'] = np.round(stimDF.loc[(0, 0), 'fNIRs offset time (s)'] / fnirs_dt)
        else:
            # Start of stimulation relative to first stimulation
            stim_start = stimDF.loc[stim, 'onset time (s)'] - stimDF.loc[(0, 0), 'onset time (s)']
            # Add time difference between current stim and first stim to fNIRs time
            stimDF.loc[stim, 'fNIRs onset time (s)'] = stimDF.loc[(0, 0), 'fNIRs onset time (s)'] + stim_start
            stimDF.loc[stim, 'fNIRs onset index'] = np.round(stimDF.loc[stim, 'fNIRs onset time (s)'] / fnirs_dt)

            # Add offset time based on duration of current stim
            stimDF.loc[stim, 'fNIRs offset time (s)'] = stimDF.loc[stim, 'fNIRs onset time (s)'] + (
                    stimDF.loc[stim, 'duration (ms)'] * 1e-3)
            stimDF.loc[stim, 'fNIRs offset index'] = np.round(stimDF.loc[stim, 'fNIRs offset time (s)'] / fnirs_dt)

    if plot:
        fig,ax = plt.subplots(figsize=(12, 8))
        plt.plot(df['Time (s)'], df['IR Filtered'].to_numpy())
        ax.axhline(y = threshold, color='r')
        plt.show()

    return stimDF

def pad_noise(array, pad_length, pad_calc_length):
    head_loc = np.mean(array[0:pad_calc_length])
    tail_loc = np.mean(array[-pad_calc_length])
    noise_scale = np.std(array)

    head_noise_array = np.random.normal(head_loc, noise_scale, pad_length)
    tail_noise_array = np.random.normal(tail_loc, noise_scale, pad_length)

    new_array = np.concatenate([head_noise_array, array, tail_noise_array])
    return new_array

def fnirs_filter(dataDF, device_type, filter_cutoffs = (0.01, 0.2), transition_width = 0.01, numtaps = 30001, freq_plot=False):

    if device_type == 'PLUX':
        chan_list = ['Red', 'IR']

        """Setup Filter"""
        fs = 1000
        filter_weights = signal.firwin(numtaps, filter_cutoffs, width=transition_width, window='Hamming', pass_zero = 'bandpass', fs=fs)

        if freq_plot:
            w,h = signal.freqz(filter_weights, worN = fft.next_fast_len(40000, real=True))
            plt.plot( (w / np.pi) * (fs/2), 20 * np.log10( np.abs(h)))
            plt.xlim((0,2))
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude (dB)')
            plt.show()

        for chan in chan_list:
            data = dataDF[chan].to_numpy()
            padded_data = pad_noise(data, numtaps, 5000)
            filtered_data = np.flip(
                signal.fftconvolve(np.flip(signal.fftconvolve(padded_data, filter_weights, mode='same')),
                                   filter_weights, mode='same'))
            name = chan + '_f'
            dataDF[name] = filtered_data[numtaps:-numtaps]

    elif device_type == 'flexNIRs':
        chan_list = ['SS Red', 'SS IR', 'D1 Ambient', 'D1 LS Red', 'D1 LL Red', 'D1 LS IR', 'D1 LL IR',
                          'D3 Ambient', 'D3 LS Red', 'D3 LL Red', 'D3 LS IR', 'D3 LL IR', 'HbT', 'SO2']

        """Setup Filter"""
        fs = 800/3
        filter_weights = signal.firwin(numtaps, filter_cutoffs, width=transition_width, window='Hamming', pass_zero = 'bandpass', fs=fs)

        if freq_plot:
            w,h = signal.freqz(filter_weights, worN = fft.next_fast_len(40000, real=True))
            plt.plot( (w / np.pi) * (fs/2), 20 * np.log10( np.abs(h)))
            plt.xlim((0,2))
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude (dB)')
            plt.show()

        for chan in chan_list:
            data = dataDF[chan].to_numpy()
            padded_data = pad_noise(data, numtaps, 5000)
            filtered_data = np.flip(
                signal.fftconvolve(np.flip(signal.fftconvolve(padded_data, filter_weights, mode='same')),
                                   filter_weights, mode='same'))
            name = chan + '_f'
            dataDF[name] = filtered_data[numtaps:-numtaps]

    return dataDF

def fnirs_plot(fnirsDF, stimDF, device_type, channel = None, pre_time = 5, post_time = 30, plot_type = 'stacked', zero_shift = False, fig_size=(10,10), legend = False, title = None, save =False):

    if device_type == 'PLUX':
        fs = 1000
        channel_pairs = {'Plux Raw' : ('Red', 'IR'),
                   'Plux Filtered' :('Red_f', 'IR_f')}

    elif device_type == 'flexNIRs':
        fs = 800 / 3
        channel_pairs = {'SS': ('SS Red', 'SS IR'),
                           'D1 LS': ('D1 LS Red', 'D1 LS IR'),
                           'D1 LL': ('D1 LL Red', 'D1 LL IR'),
                           'D3 LS': ('D3 LS Red', 'D3 LS IR'),
                           'D3 LL': ('D3 LL Red', 'D3 LL IR'),
                           'SS Filtered': ('SS Red_f', 'SS IR_f'),
                           'D1 LS Filtered': ('D1 LS Red_f', 'D1 LS IR_f'),
                           'D1 LL Filtered': ('D1 LL Red_f', 'D1 LL IR_f'),
                           'D3 LS Filtered': ('D3 LS Red_f', 'D3 LS IR_f'),
                           'D3 LL Filtered': ('D3 LL Red_f', 'D3 LL IR_f'), }

    pre_BL_idx_width = int(pre_time * fs)
    post_BL_idx_width = int(post_time * fs)

    plotDF = fnirsDF.copy()

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

    plotDF.dropna(axis = 0, subset = ['Stim #'], inplace=True)

    if plot_type == 'stacked':

        fig, ax = plt.subplots(figsize=fig_size, nrows=2)
        ax = ax.ravel()
        for idx, chan in enumerate(channel_pairs[channel]):

            # Shifts data so all trials start at 0
            if zero_shift:
                for stim_num in plotDF['Stim #'].unique():
                    zero_point = plotDF.loc[ plotDF['Stim #'] == stim_num, 'Trial Time'].abs().idxmin()
                    plotDF.loc[ plotDF['Stim #'] == stim_num, chan] = plotDF.loc[plotDF['Stim #'] == stim_num, chan] - plotDF.loc[zero_point, chan].item()

            sns.lineplot(data = plotDF, x = 'Trial Time', y = chan, hue = 'Stim #', ax = ax[idx], legend = legend)
            ax[idx].set_title(chan)
            ax[idx].set_xlabel('Time (s)')
            ax[idx].set_ylabel('A.U.')

        #This shading assumes that all stimulations within file were the same duration.  It uses the duration of the
        #last stim only
        ax[0].axvspan(xmin = 0, xmax= stimDF.loc[stim, 'duration (ms)'] * 1e-3, color = 'gray', alpha = 0.2)
        ax[1].axvspan(xmin=0, xmax=stimDF.loc[stim, 'duration (ms)'] * 1e-3, color = 'gray', alpha = 0.2)

    elif plot_type == 'average':
        fig, ax = plt.subplots(figsize=fig_size)
        for idx, chan in enumerate(channel_pairs[channel]):
            sns.lineplot(data=plotDF, x='Trial Time', y=chan, errorbar='sd', ax = ax, label = chan)
        ax.axvspan(xmin=0, xmax=stimDF.loc[stim, 'duration (ms)'] * 1e-3, color='gray', alpha=0.2)

    elif plot_type == 'Full':

        fig, ax = plt.subplots(figsize=fig_size, nrows=3, sharex=True)
        ax = ax.ravel()
        for idx, chan in enumerate(channel_pairs[channel]):
            sns.lineplot(data=plotDF, x='Trial Time', y=chan, errorbar='sd', ax = ax[0], label = chan)
        ax[0].axvspan(xmin=0, xmax=stimDF.loc[stim, 'duration (ms)'] * 1e-3, color='gray', alpha=0.2)
        ax[0].set_title(channel + ' Trial Average (mean ' + r'$\pm$' + ' s.d.)')
        ax[0].set_ylabel('A.U.')
        ax[0].legend(loc='upper right')

        for idx, chan in enumerate(channel_pairs[channel]):

            # Shifts data so all trials start at 0
            if zero_shift:
                for stim_num in plotDF['Stim #'].unique():
                    zero_point = plotDF.loc[ plotDF['Stim #'] == stim_num, 'Trial Time'].abs().idxmin()
                    plotDF.loc[ plotDF['Stim #'] == stim_num, chan] = plotDF.loc[plotDF['Stim #'] == stim_num, chan] - plotDF.loc[zero_point, chan].item()

            sns.lineplot(data = plotDF, x = 'Trial Time', y = chan, hue = 'Stim #', ax = ax[idx + 1], legend = legend)
            ax[idx + 1].set_title(chan)
            ax[idx + 1].set_xlabel('Time (s)')
            ax[idx + 1].set_ylabel('A.U.')

        #This shading assumes that all stimulations within file were the same duration.  It uses the duration of the
        #last stim only
        ax[1].axvspan(xmin = 0, xmax= stimDF.loc[stim, 'duration (ms)'] * 1e-3, color = 'gray', alpha = 0.2)
        ax[2].axvspan(xmin=0, xmax=stimDF.loc[stim, 'duration (ms)'] * 1e-3, color = 'gray', alpha = 0.2)
        if legend:
            ax[1].legend(loc='upper right')
            ax[2].legend(loc='upper right')

    fig.tight_layout()


    if title:
        fig.suptitle(title)

    if save:
        fig.savefig(title + ' ' + channel + '.png', format = 'png')

    plt.show()
    plt.close(fig)

def baseline_plot(fnirsDF, stimDF, device_type, channel, num_epochs, epoch_time, period = 'pre', plot_type = 'stacked', zero_shift = False, fig_size=(10,10), title = None, save =False):

    if device_type == 'PLUX':
        fs = 1000
        channel_pairs = {'Plux Raw' : ('Red', 'IR'),
                   'Plux Filtered' :('Red_f', 'IR_f')}

    elif device_type == 'flexNIRs':
        fs = 800 / 3
        channel_pairs = {'SS': ('SS Red', 'SS IR'),
                           'D1 LS': ('D1 LS Red', 'D1 LS IR'),
                           'D1 LL': ('D1 LL Red', 'D1 LL IR'),
                           'D3 LS': ('D3 LS Red', 'D3 LS IR'),
                           'D3 LL': ('D3 LL Red', 'D3 LL IR'),
                           'SS Filtered': ('SS Red_f', 'SS IR_f'),
                           'D1 LS Filtered': ('D1 LS Red_f', 'D1 LS IR_f'),
                           'D1 LL Filtered': ('D1 LL Red_f', 'D1 LL IR_f'),
                           'D3 LS Filtered': ('D3 LS Red_f', 'D3 LS IR_f'),
                           'D3 LL Filtered': ('D3 LL Red_f', 'D3 LL IR_f'), }


    samples_per_epoch = np.ceil(epoch_time * fs).astype(int)
    epoch_time = np.linspace(0, epoch_time, samples_per_epoch)

    """Build array indices for epochs"""
    epoch_idx_ranges = [(idx * samples_per_epoch, (idx + 1) * samples_per_epoch) for idx in np.arange(num_epochs)]

    # For pre-stim epochs -- counts backwards (e.g. index set 1 is the time immediately before stim, last set is X# epochs before stim)
    if period == 'pre':
        start_idx = stimDF.loc[(0,0), 'fNIRs onset index']
        epoch_indices = [(start_idx - idx1, start_idx - idx2) for idx1, idx2 in epoch_idx_ranges]

        #Check to for negative indices. Indicating too many or too long epochs
        for span in epoch_indices:
            for idx in span:
                if idx < 0:
                    raise Exception('Negative indices in baseline period. Reduce number or duration of epochs')

    # For post-stim epochs - counts forwards (e.g. index set 1 is time immediately after stim, last set is X# epochs after last stim
    elif period == 'post':
        start_idx = stimDF.loc[stimDF.index[-1], 'fNIRs offset index']
        epoch_indices = [(start_idx + idx1, start_idx + idx2) for idx1, idx2 in epoch_idx_ranges]

        #Check indices do not go past length of recording, Indicating too many or too long epochs
        for span in epoch_indices:
            for idx in span:
                if idx > len(fnirsDF):
                    raise Exception('Baseline period extends beyond recording. Reduce number or duration of epochs')

    """Construct plotting dataframe and plots"""
    #Construct
    plotDF = fnirsDF.copy()

    # Label data points based on stimulation #
    for idx, span in enumerate(epoch_indices):
        plotDF.loc[span[0]:span[1] - 1, "BL Epoch"] = str(idx + 1)  # -1 is because .loc slicing is endpoint inclusive
        plotDF.loc[span[0]:span[1] - 1, "Epoch Time"] = epoch_time

    plotDF.dropna(axis=0, subset=['BL Epoch'], inplace=True)

    if plot_type == 'stacked':
        fig, ax = plt.subplots(figsize=fig_size, nrows=2)
        ax = ax.ravel()
        for idx, chan in enumerate(channel_pairs[channel]):

            # Shifts data so all trials start at 0
            if zero_shift:
                for stim_num in plotDF['Stim #'].unique():
                    zero_point = plotDF.loc[ plotDF['Stim #'] == stim_num, 'Trial Time'].abs().idxmin()
                    plotDF.loc[ plotDF['Stim #'] == stim_num, chan] = plotDF.loc[plotDF['Stim #'] == stim_num, chan] - plotDF.loc[zero_point, chan].item()

            sns.lineplot(data = plotDF, x = 'Epoch Time', y = chan, hue = 'BL Epoch', ax = ax[idx])
            ax[idx].set_title(chan)
            ax[idx].set_xlabel('Time (s)')
            ax[idx].set_ylabel('A.U.')

    elif plot_type == 'average':
        fig, ax = plt.subplots(figsize=fig_size)
        for idx, chan in enumerate(channel_pairs[channel]):
            sns.lineplot(data=plotDF, x='Epoch Time', y=chan, errorbar='sd', ax = ax, label = chan)
    #
    # elif plot_type == 'Full':
    #
    #     fig, ax = plt.subplots(figsize=fig_size, nrows=3, sharex=True)
    #     ax = ax.ravel()
    #     for idx, chan in enumerate(channel_pairs[channel]):
    #         sns.lineplot(data=plotDF, x='Trial Time', y=chan, errorbar='sd', ax = ax[0], label = chan)
    #     ax[0].axvspan(xmin=0, xmax=stimDF.loc[stim, 'duration (ms)'] * 1e-3, color='gray', alpha=0.2)
    #     ax[0].set_title(channel + ' Trial Average (mean ' + r'$\pm$' + ' s.d.)')
    #     ax[0].set_ylabel('A.U.')
    #     ax[0].legend(loc='upper right')
    #
    #     for idx, chan in enumerate(channel_pairs[channel]):
    #
    #         # Shifts data so all trials start at 0
    #         if zero_shift:
    #             for stim_num in plotDF['Stim #'].unique():
    #                 zero_point = plotDF.loc[ plotDF['Stim #'] == stim_num, 'Trial Time'].abs().idxmin()
    #                 plotDF.loc[ plotDF['Stim #'] == stim_num, chan] = plotDF.loc[plotDF['Stim #'] == stim_num, chan] - plotDF.loc[zero_point, chan].item()
    #
    #         sns.lineplot(data = plotDF, x = 'Trial Time', y = chan, hue = 'Stim #', ax = ax[idx + 1])
    #         ax[idx + 1].set_title(chan)
    #         ax[idx + 1].set_xlabel('Time (s)')
    #         ax[idx + 1].set_ylabel('A.U.')
    #
    #     #This shading assumes that all stimulations within file were the same duration.  It uses the duration of the
    #     #last stim only
    #     ax[1].axvspan(xmin = 0, xmax= stimDF.loc[stim, 'duration (ms)'] * 1e-3, color = 'gray', alpha = 0.2)
    #     ax[1].legend(loc='upper right')
    #     ax[2].axvspan(xmin=0, xmax=stimDF.loc[stim, 'duration (ms)'] * 1e-3, color = 'gray', alpha = 0.2)
    #     ax[2].legend(loc='upper right')

    if title:
        fig.suptitle(title)

    if save:
        fig.savefig(title + ' ' + channel + '.png', format = 'png')

    fig.tight_layout()
    plt.show()
    plt.close(fig)

def fnirs_gigaplot(fnirsDF, stimDF, device_type, data = 'filtered', pre_time = 5, post_time = 30, zero_shift = False, fig_size=(10,10), title = None, save =False):

    if device_type == 'PLUX':
        fs = 1000
        if data == 'raw':
            channels =('Red', 'IR')
        elif data == 'filtered':
            channels = ('Red_f', 'IR_f')

    elif device_type == 'flexNIRs':
        fs = 800 / 3

        if data == 'raw':
            channels = { 'SS' : ('SS Red', 'SS IR'),
                           'D1 LS' : ('D1 LS Red', 'D1 LS IR'),
                           'D1 LL' : ('D1 LL Red', 'D1 LL IR'),
                           'D3 LS' : ('D3 LS Red', 'D3 LS IR'),
                           'D3 LL' : ('D3 LL Red', 'D3 LL IR')}
        elif data == 'filtered':
            channels = { 'SS' : ('SS Red_f', 'SS IR_f'),
                           'D1 LS' : ('D1 LS Red_f', 'D1 LS IR_f'),
                           'D1 LL' : ('D1 LL Red_f', 'D1 LL IR_f'),
                           'D3 LS' : ('D3 LS Red_f', 'D3 LS IR_f'),
                           'D3 LL' : ('D3 LL Red_f', 'D3 LL IR_f')}
        elif data == 'filtered sub SS':
            channels = { 'SS' : ('SS Red_f Normalized', 'SS IR_f Normalized'),
                           'D1 LS' : ('D1 LS Red_f Normalized sub SS', 'D1 LS IR_f Normalized sub SS'),
                           'D1 LL' : ('D1 LL Red_f Normalized sub SS', 'D1 LL IR_f Normalized sub SS'),
                           'D3 LS' : ('D3 LS Red_f Normalized sub SS', 'D3 LS IR_f Normalized sub SS'),
                           'D3 LL' : ('D3 LL Red_f Normalized sub SS', 'D3 LL IR_f Normalized sub SS')}

    pre_BL_idx_width = int(pre_time * fs)
    post_BL_idx_width = int(post_time * fs)

    plotDF = fnirsDF.copy()

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

        plotDF.loc[plot_start_idx:plot_stop_idx, 'Trial Time (s)'] = time

    plotDF.dropna(axis = 0, subset = ['Stim #'], inplace=True)

    fig, ax = plt.subplots(figsize=fig_size, nrows=3, ncols = len(channels), sharex=True, layout='constrained')
    #ax = ax.ravel()

    for col_idx, paired_channels in enumerate(channels):

        """Average Plot"""
        for idx, chan in enumerate(channels[paired_channels]):
            sns.lineplot(data=plotDF, x='Trial Time (s)', y=chan, errorbar='sd', ax = ax[0, col_idx], label = chan)
            ax[0, col_idx].set_ylabel('')

        """Stacked Plots"""
        for idx, chan in enumerate(channels[paired_channels]):

            # Shifts data so all trials start at 0
            if zero_shift:
                for stim_num in plotDF['Stim #'].unique():
                    zero_point = plotDF.loc[ plotDF['Stim #'] == stim_num, 'Trial Time (s)'].abs().idxmin()
                    plotDF.loc[ plotDF['Stim #'] == stim_num, chan] = plotDF.loc[plotDF['Stim #'] == stim_num, chan] - plotDF.loc[zero_point, chan].item()

            sns.lineplot(data = plotDF, x = 'Trial Time (s)', y = chan, hue = 'Stim #', ax = ax[idx + 1, col_idx])
            ax[idx + 1, col_idx].set_title(chan)
            #ax[idx + 1, col_idx].set_xlabel('Time (s)')
            ax[idx + 1, col_idx].set_ylabel('')

        #This shading assumes that all stimulations within file were the same duration.  It uses the duration of the
        #last stim only
        ax[0, col_idx].axvspan(xmin=0, xmax=stimDF.loc[stim, 'duration (ms)'] * 1e-3, color='gray', alpha=0.2)
        ax[0, col_idx].legend(loc='upper right')

        ax[0, col_idx].set_title(paired_channels + ' Trial Average (mean ' + r'$\pm$' + ' s.d.)')

        ax[1, col_idx].axvspan(xmin = 0, xmax= stimDF.loc[stim, 'duration (ms)'] * 1e-3, color = 'gray', alpha = 0.2)
        ax[1, col_idx].legend(loc='upper right')

        ax[2, col_idx].axvspan(xmin=0, xmax=stimDF.loc[stim, 'duration (ms)'] * 1e-3, color = 'gray', alpha = 0.2)
        ax[2, col_idx].legend(loc='upper right')

        ax[0, 0].set_ylabel('A.U.')
        ax[1, 0].set_ylabel('A.U.')
        ax[2, 0].set_ylabel('A.U.')

    ax = ax.ravel()
    for idx in np.arange(len(ax)):
        ax[idx].ticklabel_format(style='sci', axis='y', scilimits=(0, 2))

    if title:
        fig.suptitle(title)

    if save:
        fig.savefig(title + '.png', format = 'png')

    plt.show()
    plt.close(fig)

def baseline_gigaplot(fnirsDF, stimDF, device_type, num_epochs, epoch_time, period = 'pre', data = 'filtered', zero_shift = False, fig_size=(10,10), title = None, save =False):

    if device_type == 'PLUX':
        fs = 1000
        if data == 'raw':
            channels = ('Red', 'IR')
        elif data == 'filtered':
            channels = ('Red_f', 'IR_f')

    elif device_type == 'flexNIRs':
        fs = 800 / 3

        if data == 'raw':
            channels = {'SS': ('SS Red', 'SS IR'),
                        'D1 LS': ('D1 LS Red', 'D1 LS IR'),
                        'D1 LL': ('D1 LL Red', 'D1 LL IR'),
                        'D3 LS': ('D3 LS Red', 'D3 LS IR'),
                        'D3 LL': ('D3 LL Red', 'D3 LL IR')}
        elif data == 'filtered':
            channels = {'SS': ('SS Red_f', 'SS IR_f'),
                        'D1 LS': ('D1 LS Red_f', 'D1 LS IR_f'),
                        'D1 LL': ('D1 LL Red_f', 'D1 LL IR_f'),
                        'D3 LS': ('D3 LS Red_f', 'D3 LS IR_f'),
                        'D3 LL': ('D3 LL Red_f', 'D3 LL IR_f')}
        elif data == 'filtered sub SS':
            channels = { 'SS' : ('SS Red_f Normalized', 'SS IR_f Normalized'),
                           'D1 LS' : ('D1 LS Red_f Normalized sub SS', 'D1 LS IR_f Normalized sub SS'),
                           'D1 LL' : ('D1 LL Red_f Normalized sub SS', 'D1 LL IR_f Normalized sub SS'),
                           'D3 LS' : ('D3 LS Red_f Normalized sub SS', 'D3 LS IR_f Normalized sub SS'),
                           'D3 LL' : ('D3 LL Red_f Normalized sub SS', 'D3 LL IR_f Normalized sub SS')}


    samples_per_epoch = np.ceil(epoch_time * fs).astype(int)
    epoch_time = np.linspace(0, epoch_time, samples_per_epoch)

    """Build array indices for epochs"""
    epoch_idx_ranges = [(idx * samples_per_epoch, (idx + 1) * samples_per_epoch) for idx in np.arange(num_epochs)]

    # For pre-stim epochs -- counts backwards (e.g. index set 1 is the time immediately before stim, last set is X# epochs before stim)
    if period == 'pre':
        start_idx = stimDF.loc[(0,0), 'fNIRs onset index']
        epoch_indices = [(start_idx - idx1, start_idx - idx2) for idx1, idx2 in epoch_idx_ranges]

        #Check to for negative indices. Indicating too many or too long epochs
        for span in epoch_indices:
            for idx in span:
                if idx < 0:
                    raise Exception('Negative indices in baseline period. Reduce number or duration of epochs')

    # For post-stim epochs - counts forwards (e.g. index set 1 is time immediately after stim, last set is X# epochs after last stim
    elif period == 'post':
        start_idx = stimDF.loc[stimDF.index[-1], 'fNIRs offset index']
        epoch_indices = [(start_idx + idx1, start_idx + idx2) for idx1, idx2 in epoch_idx_ranges]

        #Check indices do not go past length of recording, Indicating too many or too long epochs
        for span in epoch_indices:
            for idx in span:
                if idx > len(fnirsDF):
                    raise Exception('Baseline period extends beyond recording. Reduce number or duration of epochs')

    """Construct plotting dataframe and plots"""
    #Construct
    plotDF = fnirsDF.copy()

    # Label data points based on stimulation #
    for idx, span in enumerate(epoch_indices):
        if period == 'pre':
            plotDF.loc[span[1]:span[0] - 1, "BL Epoch"] = str(idx + 1)  # -1 is because .loc slicing is endpoint inclusive
            plotDF.loc[span[1]:span[0] - 1, "Epoch Time"] = epoch_time
        elif period == 'post':
            plotDF.loc[span[0]:span[1] - 1, "BL Epoch"] = str(idx + 1)  # -1 is because .loc slicing is endpoint inclusive
            plotDF.loc[span[0]:span[1] - 1, "Epoch Time"] = epoch_time

    plotDF.dropna(axis=0, subset=['BL Epoch'], inplace=True)

    fig,ax = plt.subplots(ncols = 3 * len(channels), figsize = fig_size, layout='constrained')
    ax = ax.ravel()

    for col_idx, paired_channels in enumerate(channels):
        """Average Plot"""
        for idx, chan in enumerate(channels[paired_channels]):
            sns.lineplot(data=plotDF, x='Epoch Time', y=chan, errorbar='sd', ax = ax[col_idx * 3], label = chan)
            ax[col_idx * 3].set_title('Avg')

        # ax[0].axvspan(xmin=0, xmax=stimDF.loc[stim, 'duration (ms)'] * 1e-3, color='gray', alpha=0.2)
        # ax[0].set_title(channel + ' Trial Average (mean ' + r'$\pm$' + ' s.d.)')
        # ax[0].set_ylabel('A.U.')
        # ax[0].legend(loc='upper right')

        """Stacked Plots"""
        for idx, chan in enumerate(channels[paired_channels]):

            # # Shifts data so all trials start at 0
            # if zero_shift:
            #     for stim_num in plotDF['Stim #'].unique():
            #         zero_point = plotDF.loc[ plotDF['Stim #'] == stim_num, 'Trial Time'].abs().idxmin()
            #         plotDF.loc[ plotDF['Stim #'] == stim_num, chan] = plotDF.loc[plotDF['Stim #'] == stim_num, chan] - plotDF.loc[zero_point, chan].item()

            sns.lineplot(data = plotDF, x = 'Epoch Time', y = chan, hue = 'BL Epoch', ax = ax[(col_idx * 3) + idx + 1])

            if ((col_idx * 3) + idx + 1) % 2 == 0:
                ax[(col_idx * 3) + idx + 1].set_title('IR')
            else:
                ax[(col_idx * 3) + idx + 1].set_title('Red')
        # ax[idx + 1].set_title(chan)
        # ax[idx + 1].set_xlabel('Time (s)')
        # ax[idx + 1].set_ylabel('A.U.')

    for idx in np.arange(len(ax)):
        ax[idx].set_xlabel('')
        ax[idx].set_ylabel('')
        ax[idx].legend('').remove()
        ax[idx].ticklabel_format(style='sci', axis='y', scilimits=(0, 2))
    ax[0].set_ylabel('A.U.')

    fig.supxlabel('Baseline Epoch Time (s)')
        #print(idx)

    #This shading assumes that all stimulations within file were the same duration.  It uses the duration of the
    #last stim only
    # ax[1].axvspan(xmin = 0, xmax= stimDF.loc[stim, 'duration (ms)'] * 1e-3, color = 'gray', alpha = 0.2)
    # ax[1].legend(loc='upper right')
    # ax[2].axvspan(xmin=0, xmax=stimDF.loc[stim, 'duration (ms)'] * 1e-3, color = 'gray', alpha = 0.2)
    # ax[2].legend(loc='upper right')

    if title:
        fig.suptitle(title)

    if save:
        fig.savefig(title + '.png', format = 'png')

    plt.show()
    plt.close(fig)

def norm_sub_ss(fnirsDF):
    """Normalization and subtraction -- normalize channels against themselves.  Subtract SS channel from each channel"""
    dataDF = fnirsDF.copy()

    data_cols = ['SS Red', 'SS IR', 'D1 LS Red', 'D1 LS IR', 'D1 LL Red', 'D1 LL IR', 'D3 LS Red', 'D3 LS IR',
                 'D3 LL Red', 'D3 LL IR',
                 'SS Red_f', 'SS IR_f', 'D1 LS Red_f', 'D1 LS IR_f', 'D1 LL Red_f', 'D1 LL IR_f', 'D3 LS Red_f',
                 'D3 LS IR_f', 'D3 LL Red_f', 'D3 LL IR_f']

    reg_dct = {'SS Red Normalized': ['D1 LS Red Normalized', 'D1 LL Red Normalized', 'D3 LS Red Normalized',
                                     'D3 LL Red Normalized'],
               'SS IR Normalized': ['D1 LS IR Normalized', 'D1 LL IR Normalized', 'D3 LS IR Normalized',
                                    'D3 LL IR Normalized'],
               'SS Red_f Normalized': ['D1 LS Red_f Normalized', 'D1 LL Red_f Normalized', 'D3 LS Red_f Normalized',
                                       'D3 LL Red_f Normalized'],
               'SS IR_f Normalized': ['D1 LS IR_f Normalized', 'D1 LL IR_f Normalized', 'D3 LS IR_f Normalized',
                                      'D3 LL IR_f Normalized'], }

    for column in data_cols:
        norm_col_title = column + ' Normalized'
        norm_val = np.max(dataDF[column])
        dataDF[norm_col_title] = dataDF[column].div(norm_val)

    for k, v in reg_dct.items():
        for col in v:
            subtraction_col_title = col + ' sub SS'
            dataDF[subtraction_col_title] = dataDF[col] - dataDF[k]

    return dataDF
