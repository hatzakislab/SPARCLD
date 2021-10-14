#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 11:10:42 2020

@author: mettemalle
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:40:36 2020

@author: mettemalle
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 16:53:04 2018

@author: mettemalle
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy import fftpack
import scipy
from scipy.signal import find_peaks, peak_prominences

from scipy.signal import butter, lfilter, freqz
from pathlib import Path
from typing import Union, Tuple, List, AnyStr
import xgboost as xgb
from sklearn.base import ClassifierMixin, TransformerMixin
from pandas import DataFrame
import pickle
from sklearn.preprocessing import LabelEncoder
from itertools import combinations
from scipy import signal
from scipy.signal import butter,filtfilt


def load_model(model_path: Union[Path, str],
               ) -> Tuple[Union[xgb.XGBClassifier, ClassifierMixin], TransformerMixin]:
    """
    Loads model from path, and checks it before passing it on
    :param model_path: Path object from which to load the model. must end with '.model'
    :return:
    """
    assert isinstance(model_path, (Path, str))
    if type(model_path) == str:
        model_path = Path(model_path)
    # ensure that the model path is the right extension
    assert str(model_path).endswith('.model')
    assert model_path.exists()
    model = pickle.load(open(str(model_path), 'rb'))
    check_model(model)
    return model



def check_model(model: object) -> None:
    """
    checks whether the model is correct
    """
    clf, enc = model
    assert isinstance(clf, (xgb.XGBClassifier, ClassifierMixin))
    assert isinstance(enc, (TransformerMixin, LabelEncoder))
    pass

def predict_signatures(model, X_df):
    (clf, enc) = model

    y_pred = clf.predict(X_df)
    y_labels = enc.inverse_transform(y_pred)

    return y_labels

def get_X_from_df(df: DataFrame, clf: Union[xgb.XGBClassifier, ClassifierMixin]) -> DataFrame:
    cols_all, _ = get_signal_columns(df)

    X = df[cols_all]

    X = preprocess_X(X)

    X = add_missing_columns(X, clf)

    return X


def predict_signatures(model, X_df):
    (clf, enc) = model

    y_pred = clf.predict(X_df)
    y_labels = enc.inverse_transform(y_pred)

    return y_labels


def preprocess_X(df: DataFrame,
                 log: bool = True,
                 ratios: bool = True,
                 ) -> DataFrame:
    """
    Adds columns and more, depending on the boolean switches
    :param df:
    :param log:bool, whether to include log values of the different colors
    :param ratios:bool, whether to include ratios between the different colors
    :return df : preprocessed
    """
    # Start by bg correcting (if it has been done already, nothing should happen here)
    df = df.copy()
    #df = ld.do_bg_correction(df)     #it is bg corrected
    _, colors = get_signal_columns(df)

    colors = sorted(colors, reverse=True)

    if log:
        for col in colors:
            logcol = f'{col}_log'
            df[logcol] = np.log10(np.clip(df[col], 0.0001, max(df[col])))
    if ratios:
        for col_a, col_b in combinations(colors, 2):
            df[f"ratio_{col_a}_{col_b}"] = df[col_a] / df[col_b]

    if df.isna().any().any():
        df.fillna(value=0, inplace=True)

    return df

raw_cols = [
    'red',
    'green',
    'blue',
    'red_bg',
    'green_bg',
    'blue_bg',
]

def add_missing_columns(X, clf) -> DataFrame:
    model_cols = clf.get_booster().feature_names
    missing_cols = [col for col in model_cols if col not in X.keys()]
    X = X.copy()
    if len(missing_cols) != 0:
        for col in missing_cols:
            X[col] = add_column(X, col)
    X = X[model_cols]  # ensure the same order of columns. Will throw error with missing cols
    return X

def get_signal_columns(df: pd.DataFrame) -> tuple:
    """
    removes private columns, and returns signal (rw + bg ) and raw column names.
    :param df:
    :return cols_all: signal column names
    :return cols_colors: raw (excluding bg) column names
    """
    # return cols, but remove "private" cols, those that shouldn't be read in the peakfinder
    cols = list(df.keys())
    cols_all = [col for col in cols if col in raw_cols]
    cols_colors = [col for col in cols_all if not col.endswith('_bg')]
    return cols_all, cols_colors


def smoothpower(dy, p):
    ps_dy = np.power(dy, p)
    return ps_dy

def peakfinder(ps_d, sigma_d, distance, prom):
    mean = np.mean(ps_d)
    std = np.std(ps_d)
    peak1, properties = scipy.signal.find_peaks(ps_d, height = mean + sigma_d*std, distance = distance, prominence = prom)
    return peak1

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y
    
        
def normalize(s):
    return (s - s.min()) / (s.max() - s.min())

def unique(list1): 
    x = np.array(list1) 
    return (np.unique(x))


#partikel ID

def Meantwodiff(t,v):
        """Differentiates a signal two times and returns time and d^2/dv^2"""
        dy=np.diff(v,1)
        dx=np.diff(t,1)
        yfirst=dy/dx
        xfirst=0.5*(t[:-1]+t[1:])
        dyfirst=np.diff(yfirst,1)
        dxfirst=np.diff(xfirst,1)
        ysecond=dyfirst/dxfirst
        xsecond=0.5*(xfirst[:-1]+xfirst[1:])

        return xfirst,yfirst,xsecond,ysecond
    
def peakfinder_kr(ps_d, ps_dd, sigma_d, sigma_dd):
    mean = np.mean(ps_d)
    std = np.std(ps_d)
    peak1, properties = scipy.signal.find_peaks(ps_d, height = mean + sigma_d*std)
    mean2 = np.mean(ps_dd)
    std2 = np.std(ps_dd)
    peak2, properties2 = scipy.signal.find_peaks(ps_dd, height = mean2 + sigma_dd*std2)

    return peak1, peak2

def smoothpower_double(dy,ddy,p):
    
    ps_dy = np.power(dy, p)
    ps_ddy = np.power(ddy, p)
    
    return ps_dy, ps_ddy
def smoothdif(dy, ddy, s):
    smooth_dy = scipy.ndimage.gaussian_filter(dy, sigma = s)
    smooth_ddy = scipy.ndimage.gaussian_filter(dy, sigma = s)
    return smooth_dy, smooth_ddy

def smooth(dy, s):
    smooth_dy = scipy.ndimage.gaussian_filter(dy, sigma = s)

    return smooth_dy


# label 
#number = np.arange(6)

number = np.arange(6)


for i in number: 
    particleID = []

    peak_prominences_list = []
    peak_prominences_neg_list = []

    counter = []
    N_jump_off = []
    N_peaks = []

    N_peaks_time = []
    N_jump_off_time = []


    N_peaks_red = []
    N_peaks_green = []
    N_peaks_blue = []
    N_peaks_time_red = []
    N_peaks_time_green = []
    N_peaks_time_blue = []

    increase_green = []
    increase_red = []
    increase_blue = []

    predicted_lina = []
    
    
    
    particleID_kr = []
    counter_kr = []     
    N_peaks_time_kr = []  
    N_peak_kr = []
         
    increase_red_kr = []
    increase_green_kr = []
    increase_blue_kr = []
    
    predicted_lina_kr = []
        
    
    target_int = []
    target_int_kr = []
    
    
    number = i
    print('round..............................')
    print(number)
    PATH = '/Users/mettemalle/Nextcloud/DNA Fusion assay/magnus data/fusion/TIFs3/'
    
    data = pd.read_csv(PATH + str(number) + "/signal_df.csv", dtype={'user_id': int})
    
    #data = data[(data.particle == 0)]
    
    for name, particle in data.groupby('particle'):
        
        norm = particle.loc[:, ['red_int_corrected', 'green_int_corrected', 'blue_int_corrected']]
       
        norm = norm.apply(normalize, axis=0)
        
        norm['total'] = norm.apply(np.sum, axis=1)
        
        norm['total'] = normalize(norm.loc[:, 'total'])
        
        norm['total'] = norm.loc[:, 'total'] - norm.loc[:, 'total'].mean()
    
        particle = particle.join(norm, rsuffix="_norm")
        
        
        #mass = particle['mass']
        
        sig = particle['total']
        sig_fft = fft(sig)
  #-----------------------------------------------------------------------_#

       #finding kiss and runs 
        
        kr_red = particle['red_int_corrected_norm']
        kr_green = particle['green_int_corrected_norm']
        kr_blue = particle['blue_int_corrected_norm']
        kr_multi = [sum(x) for x in zip(kr_red, kr_green, kr_blue)]
        t = np.arange(len(kr_red))
        
        
        # Filter requirements.
        order = 1    
        fs = 31.0       # sample rate, Hz
        cutoff = 1  # desired cutoff frequency of the filter, Hz
        nyq = 0.5 * fs
        # Get the filter coefficients so we can check its frequency response.
        b, a = butter_lowpass(cutoff, fs, order)
        # Filter the data, and plot both the original and filtered signals.
        Filtered_sig = butter_lowpass_filter(sig, cutoff, fs, order)
    
        #kr_red = butter_lowpass_filter(kr_red, cutoff, fs, order)
        #kr_green = butter_lowpass_filter(kr_green,  cutoff, fs, order)
        #kr_blue = butter_lowpass_filter(kr_blue, cutoff, fs, order)  
        
        
        sigma = 0 
        # differentiate signal
        xfirstG,yfirstG,xsecondG,ysecondG = Meantwodiff(t,kr_green)
        xfirstB,yfirstB,xsecondB,ysecondB = Meantwodiff(t,kr_blue)
        xfirstR,yfirstR,xsecondR,ysecondR = Meantwodiff(t,kr_red)
        xfirstM,yfirstM,xsecondM,ysecondM = Meantwodiff(t,kr_multi)
        
        green_dy, green_ddy = smoothdif(yfirstG,ysecondG,sigma) 
        blue_dy, blue_ddy = smoothdif(yfirstB,ysecondB,sigma)
        red_dy, red_ddy = smoothdif(yfirstR,ysecondR,sigma)  
        
        green_ps_dy, green_ps_ddy = smoothpower_double(green_dy, green_ddy,3)
        blue_ps_dy, blue_ps_ddy = smoothpower_double(blue_dy, blue_ddy,3)
        red_ps_dy, red_ps_ddy = smoothpower_double(red_dy, red_ddy,3)
    
        # laver en multichannel
        multi_dy_kr = [sum(x) for x in zip(green_dy, red_dy, blue_dy)]
        
        multi_ddy_kr = [sum(x) for x in zip(green_ddy, red_ddy, blue_ddy)]
    
        std_multi = np.std(kr_multi)
        std_green = np.std(kr_green)
        std_red = np.std(kr_red)
        std_blue = np.std(kr_blue)
        # finder peaks kiss and runs
        green_peak1_kr, green_peak2_kr = peakfinder_kr(green_ps_dy, green_ps_ddy, 0.7, 0.8)
        blue_peak1_kr, blue_peak2_kr = peakfinder_kr(blue_ps_dy, blue_ps_ddy, 0.9, 1)
        red_peak1_kr, red_peak2_kr = peakfinder_kr(red_ps_dy, red_ps_ddy, 0.7 , 0.8)
        #multi_peak1_kr, multi_peak2_kr = peakfinder_kr(multi_dy_kr, multi_ddy_kr, 2.3, 2.3)
        
        # check for concistent intensity increase 
        peak_threshold_kr = 1
        peak_threshold_after = 1
    
        forward = 10
        startframe = 3  
        kissnrun_final_red = []
        kissnrun_final_green = []
        kissnrun_final_blue = []
        
        for peak in range(len(red_peak1_kr)):
             
            if std_red < 0.16:
                kissnrun_final_red.append(red_peak1_kr[peak])
    
        for peak in range(len(green_peak1_kr)):
             
            if std_green < 0.16:
                kissnrun_final_green.append(green_peak1_kr[peak])
        
        for peak in range(len(blue_peak1_kr)):
            
            if std_blue < 0.15:
                kissnrun_final_blue.append(blue_peak1_kr[peak])
                
        
            xvalues_red_kr = xfirstR[kissnrun_final_red]+0.5  
            xvalues_green_kr = xfirstG[kissnrun_final_green]+0.5  
            xvalues_blue_kr = xfirstB[kissnrun_final_blue]+0.5  
        
            dy_kr = np.asarray(multi_dy_kr)
        
            yvalues_red_kr = dy_kr[kissnrun_final_red]
            yvalues_green_kr = dy_kr[kissnrun_final_green]
            yvalues_blue_kr = dy_kr[kissnrun_final_blue]
        
                
        combined_x_kr = unique((list(xvalues_red_kr) + list(xvalues_green_kr) + list(xvalues_blue_kr)))
        combined_y_kr = unique((list(yvalues_red_kr) + list(yvalues_green_kr) + list(yvalues_blue_kr)))
        
  
 #-------------------------------------------------------------------------------#   
        # Filter requirements.
        order = 1
        fs = 40.0       # sample rate, Hz
        cutoff = 15  # desired cutoff frequency of the filter, Hz
        # Get the filter coefficients so we can check its frequency response.
        b, a = butter_lowpass(cutoff, fs, order)
        # Filter the data, and plot both the original and filtered signals.
        Filtered_sig = butter_lowpass_filter(sig, cutoff, fs, order)
    
        Filtered_red = butter_lowpass_filter(particle['red_int_corrected_norm'], cutoff, fs, order)
        Filtered_green = butter_lowpass_filter(particle['green_int_corrected_norm'], cutoff, fs, order)
        Filtered_blue = butter_lowpass_filter(particle['blue_int_corrected_norm'], cutoff, fs, order)   
        
        # step finding algorithm
        L = 20   
        step = np.hstack((0.5*np.ones(L), -0.5*np.ones(L)))
        particle_step = np.convolve(norm['total'], step, mode='same')
        # step algorith på fitlered data
        particle_step_filter = np.convolve(sig, step, mode='same')
        
        Particle_step_filter_red = np.convolve(particle['red_int_corrected_norm'], step, mode='same')
        Particle_step_filter_green = np.convolve(particle['green_int_corrected_norm'], step, mode='same')
        Particle_step_filter_blue = np.convolve(particle['blue_int_corrected_norm'], step, mode='same')
        
        
        # peak detection algorithm 
        
        #filtered step trace in power function
        particle_step_filter_power = smoothpower(particle_step_filter, 3)
        particle_step_filter_red_power = smoothpower(Particle_step_filter_red, 3)
        particle_step_filter_green_power = smoothpower(Particle_step_filter_green, 3)
        particle_step_filter_blue_power = smoothpower(Particle_step_filter_blue, 3)
        
        
        time_length = max(particle['frame'])
        
        #peakfinding
        Step_peak_1 = peakfinder(particle_step_filter_power, 0.2, 10, 0.5)
        step_peak = []
        for peaks in Step_peak_1:
            if peaks > 10 and peaks < (time_length - 10):
                step_peak.append(peaks)
        
        
        Step_peak_1red = peakfinder(particle_step_filter_red_power, 1, 10, 5)
        step_peak_red = []
        for peaks in Step_peak_1red:
            if peaks > 10 and peaks < (time_length - 10):
                step_peak_red.append(peaks)
        Step_peak_1green = peakfinder(particle_step_filter_green_power, 1, 10, 5)
        step_peak_green = []
        for peaks in Step_peak_1green:
            if peaks > 10 and peaks < (time_length - 10):
                step_peak_green.append(peaks)
        Step_peak_1blue = peakfinder(particle_step_filter_blue_power, 1, 10, 5)
        step_peak_blue = []
        for peaks in Step_peak_1blue:
            if peaks > 10 and peaks < (time_length - 10):
                step_peak_blue.append(peaks)
        
        
        #step_peak = np.asarray(step_peak)
        prominences = peak_prominences(particle_step_filter_power, step_peak)[0]
        prominences_red = peak_prominences(particle_step_filter_red_power, step_peak_red)[0]
        prominences_green = peak_prominences(particle_step_filter_green_power, step_peak_green)[0]
        prominences_blue = peak_prominences(particle_step_filter_blue_power, step_peak_blue)[0]
        
        
        # combining peaks and get intensities 
        combined_peak = unique(step_peak_red + step_peak_green + step_peak_blue)
        
        int_red = []
        int_green = []    
        int_blue = []
        red_int_tmp =  particle['red_int_corrected'].tolist()
        green_int_tmp =  particle['green_int_corrected'].tolist()
        blue_int_tmp =  particle['blue_int_corrected'].tolist()
        
        for i in combined_peak:
            red = np.mean(red_int_tmp[int(i)+1:int(i)+6])-np.mean(red_int_tmp[int(i)-6:int(i)-1])
            green = np.mean(green_int_tmp[int(i)+1:int(i)+6])-np.mean(green_int_tmp[int(i)-6:int(i)-1])
            blue = np.mean(blue_int_tmp[int(i)+1:int(i)+6])-np.mean(blue_int_tmp[int(i)-6:int(i)-1])
            int_red.append(red)
            int_green.append(green)
            int_blue.append(blue)
    
        # intensities for kiss and runs 
        int_red_kr = []
        int_green_kr = []    
        int_blue_kr = []
        
        for i in combined_x_kr:
            red_kr = red_int_tmp[int(i)] - 0.8*np.mean(red_int_tmp[int(i-30):int(i+30)])
            green_kr = green_int_tmp[int(i)] - 1*np.mean(green_int_tmp[int(i-30):int(i+30)])
            blue_kr = blue_int_tmp[int(i)] - 1.1*np.mean(blue_int_tmp[int(i-30):int(i+30)])
            int_red_kr.append(red_kr)
            int_green_kr.append(green_kr)
            int_blue_kr.append(blue_kr)

        mass = np.mean(blue_int_tmp[0:5])
        #peak_prominences_list.append(prominences)
        
        # off jumping finding
        particle_step_filter_power_neg = [0 - x for x in particle_step_filter_power]
        Neg_step_peak_1 = peakfinder(particle_step_filter_power_neg, 0.5, 10, 5)
        neg_step_peak = []
        for peaks in Neg_step_peak_1:
            if peaks > 10 and peaks < (time_length - 10):
                neg_step_peak.append(peaks)
                
        particle_step_filter_power_neg_red = [0 - x for x in particle_step_filter_red_power]
        Neg_step_peak_1red = peakfinder(particle_step_filter_power_neg_red, 0.5, 10, 5)
        neg_step_peak_red = []
        for peaks in Neg_step_peak_1red:
            if peaks > 10 and peaks < (time_length - 10):
                neg_step_peak_red.append(peaks)    
        particle_step_filter_power_neg_green = [0 - x for x in particle_step_filter_green_power]
        Neg_step_peak_1green = peakfinder(particle_step_filter_power_neg_green, 0.5, 10, 5)
        neg_step_peak_green = []
        for peaks in Neg_step_peak_1green:
            if peaks > 10 and peaks < (time_length - 10):
                neg_step_peak_green.append(peaks)
        particle_step_filter_power_neg_blue = [0 - x for x in particle_step_filter_blue_power]
        Neg_step_peak_1blue = peakfinder(particle_step_filter_power_neg_blue, 0.5, 10, 5)
        neg_step_peak_blue = []
        for peaks in Neg_step_peak_1blue:
            if peaks > 10 and peaks < (time_length - 10):
                neg_step_peak_blue.append(peaks)
                
        prominences_neg = peak_prominences(particle_step_filter_power_neg, neg_step_peak)[0]
        
        peak_prominences_neg_list.append(prominences_neg)
        
        # combining peaks and get intensities 
        combined_peak_neg = unique(neg_step_peak_red + neg_step_peak_green + neg_step_peak_blue)
       
        #deleting noisy and bad traces
        
        if len(combined_peak_neg) > 0 and len(combined_peak) == 0:
            combined_peak = []
            combined_peak_neg = []
        
        elif len(combined_peak_neg) > 0 and len(combined_peak) > 0 and combined_peak_neg[0] < combined_peak[0]:
            combined_peak = []
            combined_peak_neg = []
        
            
            
        # plotting    
        
        x = np.asarray(particle['frame'])
        xvalues = x[step_peak]  
        particle_step_filter = np.asarray(particle_step_filter)
        yvalues = particle_step_filter[step_peak]
        
        xnegative = x[neg_step_peak]  
        ynegative = particle_step_filter[neg_step_peak]
        
        signal = np.asarray(sig)
        yvalues_signal = signal[step_peak]
        ynegative_signal = signal[neg_step_peak]
        
        yvalues_signal_red = signal[step_peak_red]
        yvalues_signal_green = signal[step_peak_green]
        yvalues_signal_blue = signal[step_peak_blue]
        ynegative_signal_red = signal[neg_step_peak_red]
        ynegative_signal_green = signal[neg_step_peak_green]
        ynegative_signal_blue = signal[neg_step_peak_blue]
        
        ypower = particle_step_filter_power[step_peak]
        ynegative_power= particle_step_filter_power[neg_step_peak]
        
        xvalues_red = x[step_peak_red] 
        power_red = particle_step_filter_red_power[step_peak_red]
        xvalues_green = x[step_peak_green] 
        power_green = particle_step_filter_green_power[step_peak_green]
        xvalues_blue = x[step_peak_blue] 
        power_blue = particle_step_filter_blue_power[step_peak_blue]
        
        xnegative_red = x[neg_step_peak_red]
        xnegative_green = x[neg_step_peak_green]  
        xnegative_blue = x[neg_step_peak_blue]  
        ynegative_power_red = particle_step_filter_red_power[neg_step_peak_red]
        ynegative_power_green = particle_step_filter_green_power[neg_step_peak_green]
        ynegative_power_blue = particle_step_filter_blue_power[neg_step_peak_blue]
        
        plt.ioff()
        fig, ax = plt.subplots(4,1,figsize=(8,12),sharex=True)
        
       
        ax[0].plot(particle['frame'], particle['green_int_corrected_norm'], label='532 signal', color = 'limegreen')
        ax[0].plot(particle['frame'], particle['blue_int_corrected_norm'], label='488 signal', color = 'cornflowerblue')
        
        ax[0].plot(particle['frame'], particle['total'], label='multi signal', color = 'darkmagenta')   
        ax[0].plot(particle['frame'], particle['red_int_corrected_norm'], label='655 signal', color = 'salmon')   
        #ax[0].legend()
        ax[0].set(title="Raw")   
        
        ax[0].plot(combined_x_kr, combined_y_kr, "x", color = 'darkgreen')
        #ax[0].set_xlim(430, 490)

        
        #ax[1].plot(particle['frame'], particle_step_filter_red, label='filtered red signal convolution', color = 'red')
        #ax[1].plot(particle['frame'], particle_step_filter_green, label='filtered green signal convolution', color = 'green')
        #ax[1].plot(particle['frame'], particle_step_filter_blue, label='filtered blue signal convolution', color = 'blue')
        
        ax[1].plot(particle['frame'][0:-1], green_ps_dy, label='filtered blue signal convolution', color = 'green')
        ax[1].plot(particle['frame'][0:-1], red_ps_dy, label='filtered blue signal convolution', color = 'red')
        ax[1].plot(particle['frame'][0:-1], blue_ps_dy, label='filtered blue signal convolution', color = 'blue')
        
        ax[1].plot(xvalues_red_kr, yvalues_red_kr, "x", color = 'red')
        ax[1].plot(xvalues_green_kr, yvalues_green_kr, "x", color = 'green')
        ax[1].plot(xvalues_blue_kr, yvalues_blue_kr, "x", color = 'blue')
        
    
        ax[2].plot(particle['frame'], particle_step_filter_red_power, color = 'red')
        ax[2].plot(xvalues_red, power_red, "x", color = 'maroon')
        ax[2].plot(particle['frame'], particle_step_filter_green_power, color = 'green')
        ax[2].plot(xvalues_green, power_green, "x", color = 'darkgreen')
        ax[2].plot(particle['frame'], particle_step_filter_blue_power, color = 'blue')
        ax[2].plot(xvalues_blue, power_blue, "x", color = 'navy')
        
        ax[2].plot(xnegative_red, ynegative_power_red, "d", color = 'maroon')
        ax[2].plot(xnegative_green, ynegative_power_green, "d", color = 'darkgreen')
        ax[2].plot(xnegative_blue, ynegative_power_blue, "d", color = 'navy')
        
        ax[2].set(title="individual power signals convoled with a step")
    
        
        ax[3].plot(particle['frame'], kr_red, label='raw signal', color = 'red')
    
        ax[3].plot(particle['frame'], kr_blue, label='filtered signal', color = 'blue')
        ax[3].plot(particle['frame'], kr_green, label='filtered signal', color = 'green')
        ax[3].set(title="Low pass filtering")
        ax[3].plot(xvalues_red, yvalues_signal_red, "x", color = 'maroon')
        ax[3].plot(xvalues_green, yvalues_signal_green, "x", color = 'darkgreen')
        ax[3].plot(xvalues_blue, yvalues_signal_blue, "x", color = 'navy')
        
        ax[3].plot(xnegative_red, ynegative_signal_red, "d", color = 'maroon')
        ax[3].plot(xnegative_green, ynegative_signal_green, "d", color = 'darkgreen')
        ax[3].plot(xnegative_blue, ynegative_signal_blue, "d", color = 'navy')
        
        fig.savefig(PATH + '/peaks/' + str(number) + '_' + str(name) + '_peak_finder.pdf')
        plt.close('all')
        
        N_jump_off_time1 = []
        
        if len(neg_step_peak) > 0: 
            N_jump_off_time1.append(neg_step_peak[0])
        else: 
            N_jump_off_time1.append(1200)  
            
        N_jump_off_time2 = np.asarray(N_jump_off_time1[0])
        
        #print(combined_peak)
        
        
        # predicting classes
    
        # use model :  library_001_105_011_101.model
        
        int_red = int_red
        int_green = int_green
        int_blue = int_blue
        
        combined_int = []
        if len(combined_peak) > 0: 
            d = {'red': int_red, 'green': int_green, 'blue': int_blue}
            combined_int.append(pd.DataFrame(data=d))
            #combined_int.append(pd.DataFrame({'red':[np.asarray(int_red)],'green':[np.asarray(int_green)],'blue':[np.asarray(int_blue)]}))
        
        d = {'red': int_red, 'green': int_green, 'blue': int_blue}
        combined_int = pd.DataFrame(data=d)
        
        # combine data if not empty ! 
        

        
        # kiss and run data 
        int_red_kr = int_red_kr
        int_green_kr = int_green_kr
        int_blue_kr = int_blue_kr
        
        combined_int_kr = []
        if len(combined_x_kr) > 0: 
            d_kr = {'red': int_red_kr, 'green': int_green_kr, 'blue': int_blue_kr}
            combined_int_kr.append(pd.DataFrame(data=d_kr))
            #combined_int.append(pd.DataFrame({'red':[np.asarray(int_red)],'green':[np.asarray(int_green)],'blue':[np.asarray(int_blue)]}))
        d_kr = {'red': int_red_kr, 'green': int_green_kr, 'blue': int_blue_kr}
        combined_int_kr = pd.DataFrame(data=d_kr)        
        
        # loading model
        model_dir = Path('/Users/mettemalle/Nextcloud/DNA Fusion assay/20191218/model_4')
        model_path = sorted(list(model_dir.glob('library4_001_010_100_111_105_501.model')))[0]
    
        model = load_model(model_path)
        clf, enc = model
        
        # center data frame for prediction combined_int
        lina = []
        if len(combined_int) > 0:
            centered_data = get_X_from_df(combined_int,clf)
        
        # predicting the label 
        
            y_labels = predict_signatures(model, centered_data)
            
            print(y_labels)
        
        
        
        #predict kiss and run event
        lina_kr = []
        if len(combined_int_kr) > 0:
            centered_data_kr = get_X_from_df(combined_int_kr,clf)
        
        # predicting the label 
        
            y_labels_kr = predict_signatures(model, centered_data_kr)
            
            print(y_labels_kr)
        
        #y_labels = pd.DataFrame(data=y_labels_test)
        #print(y_labels2)
        
        
        # translate predicted label
        lina = []
        for labels in y_labels: 
            
            if labels == '1:0:0':          
                lina.append('A')
            if labels == '0:1:0':
                lina.append('B')
            if labels == '0:0:1':
                lina.append('C')
            if labels == '1:1:1':
                lina.append('D')
            if labels == '1:0:5':
                lina.append('E')
            if labels == '5:0:1':
                lina.append('F')
       
        #lina_kr = []
        #for labels in y_labels_kr: 
            
        #    if labels == '1:0:0':          
        #        lina_kr.append('A')
        #    if labels == '0:1:0':
        #        lina_kr.append('B')
        #    if labels == '0:0:1':
        #        lina_kr.append('C')
        #    if labels == '1:1:1':
        #        lina_kr.append('D')
        #    if labels == '1:0:5':
        #        lina_kr.append('E')
        #    if labels == '5:0:1':
        #        lina_kr.append('F')
       
        
        
        
        if len(combined_peak) > 0: 
            
        
            for event in range(len(combined_peak)):
            
                particleID.append(name + 1000*number) 
                counter.append(event)
                target_int.append(mass)
                
                N_peaks.append(len(combined_peak))
                N_jump_off.append(len(combined_peak_neg))
               
                N_peaks_time.append(combined_peak[event])
               
                increase_red.append(int_red[event])
                increase_green.append(int_green[event])
                increase_blue.append(int_blue[event])
               
                predicted_lina.append(lina[event])
                
        """        
        if len(combined_int_kr) > 0 and len(combined_x_kr) < 10: 
            
            
            for event in range(len(combined_int_kr)):
            
                particleID_kr.append(name + 1000*number) 
                counter_kr.append(event)
                target_int_kr.append(mass)
                
                predicted_lina_kr.append(lina_kr[event])
            
                increase_red_kr.append(int_red_kr[event])
                increase_green_kr.append(int_green_kr[event])
                increase_blue_kr.append(int_blue_kr[event])
                    
                N_peak_kr.append(len(combined_x_kr))
                N_peaks_time_kr.append(combined_x_kr[event])
         """       
                #predicted
                
        # predicted label
    
    final_df = pd.DataFrame(
        {'particle_ID': particleID,
    
         'event_number': counter, 
         
         'N_peaks_time': N_peaks_time,
         
         'N_peaks': N_peaks,
         
         'N_jump_off': N_jump_off,
         
         'I_red': increase_red,
         
         'I_green': increase_green,
         
         'I_blue': increase_blue,
              
         'lina_label': predicted_lina, 
         
         'target_liposome' : target_int,
         })
    
    final_df.to_csv(str(PATH + str(number) + '_target_peak_results.csv'), header=True, index=None, sep=',', mode='w')
             
           
##---------------------------------------------------------------#
                
                   
                #predicted
                
        # predicted label
    
    #final_df_kr = pd.DataFrame(
    #    {'particle_ID_kr': particleID_kr,
   # 
    #     'event_number': counter_kr, 
         
    #     'N_peaks_time': N_peaks_time_kr,
         
     #    'N_peaks_kr': N_peak_kr,
         
       #  'I_red_kr': increase_red_kr,
      #   
        # 'I_green_kr': increase_green_kr,
         
         #'I_blue_kr': increase_blue_kr,
              
         #'lina_label_kr': predicted_lina_kr, 
         
        # 'target_liposome' : target_int_kr,
        # })
    
  #  final_df_kr.to_csv(str(PATH + str(number) + '_target_kiss_and_run_results.csv'), header=True, index=None, sep=',', mode='w')
        
