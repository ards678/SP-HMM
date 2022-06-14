from distutils.command.upload import upload
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import noisereduce as nr
from pathlib import Path
from scipy.io import wavfile
import librosa as lb
import librosa.display
import pyAudioAnalysis as pyaudio
import IPython, itertools
import os, pickle, csv
import random, re
import ipywidgets as widgets
import streamlit as st

from glob import glob
from pyAudioAnalysis import ShortTermFeatures as aSF
from pyAudioAnalysis import MidTermFeatures as aMF
from pyAudioAnalysis import audioBasicIO as aIO
from random import shuffle
from math import floor

from IPython.display import Audio
from IPython.display import display

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from hmmlearn import hmm

#Class to handle all HMM related processing
class HMMTrainer(object):
    def __init__(self, n_components=7, model_name ="GaussianHMM", cov_type='tied', n_iter=10000):
        self.model_name = model_name
        self.n_components = n_components
        self.cov_type = cov_type
        self.n_iter = n_iter
        self.models = []

        if self.model_name == 'GaussianHMM':
            self.model = hmm.GaussianHMM(n_components=self.n_components, covariance_type=self.cov_type, n_iter=self.n_iter)
        elif self.model_name == 'GMMHMM':
            self.model = hmm.GMMHMM(n_components=self.n_components, covariance_type=self.cov_type, n_iter=self.n_iter)
        elif self.model_name == 'MultinomialHMM':
            self.model = hmm.MultinomialHMM(n_components=self.n_components, n_iter=self.n_iter)
        else:
            raise TypeError("Invalid model type")

    # X is a 2D numpy array where each row is 13D
    def train(self, X):
        np.seterr(all='ignore')
        self.models.append(self.model.fit(X))

    # Run model on input data
    def get_score(self, input_data):
        return self.model.score(input_data)
    
    def get_predict(self, X):
        return self.model.predict(X, lengths=None)

#Noise Reduction via Spectral Grating
def noise_reduction_stationary(data, rate):
    reduced_noise = nr.reduce_noise(y=data, sr=rate, n_std_thresh_stationary=1.5, stationary=True)
    return reduced_noise

def noise_reduction_nonstationary(data, rate):
    reduced_noise = nr.reduce_noise(y=data, sr=rate, thresh_n_mult_nonstationary=1.25, stationary=False)
    return reduced_noise

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def get_normalized_features(df):
    minim = data_summary.loc["min"]
    minim = minim.to_numpy()
    minim = minim.reshape(-1,1)
    minim = minim.T
    maxim = data_summary.loc["max"]
    maxim = maxim.to_numpy()
    maxim = maxim.reshape(-1,1)
    maxim = maxim.T
    norm_features = (df-minim)/ (maxim-minim)
    norm_features
    
    return norm_features

def Filter(string, substr):
    return [str for str in string if
             any(sub in str for sub in substr)]

global s, fs, max_score, output_label, hmm_models

hmm_models = []

angHMM = pickle.load(open("Complete_ANGRY_model2.pkl", 'rb'))
hmm_models.append((angHMM, 'ANGRY'))

disHMM = pickle.load(open("Complete_DISGUST_model2.pkl", 'rb'))
hmm_models.append((disHMM, 'DISGUST'))

feaHMM = pickle.load(open("Complete_FEAR_model2.pkl", 'rb'))
hmm_models.append((feaHMM, 'FEAR'))

hapHMM = pickle.load(open("Complete_HAPPY_model2.pkl", 'rb'))
hmm_models.append((hapHMM, 'HAPPY'))

neuHMM = pickle.load(open("Complete_NEUTRAL_model2.pkl", 'rb'))
hmm_models.append((neuHMM, 'NEUTRAL'))

sadHMM = pickle.load(open("Complete_SAD_model2.pkl", 'rb'))
hmm_models.append((sadHMM, 'SAD'))

surHMM = pickle.load(open("Complete_SURPRISE_model2.pkl", 'rb'))
hmm_models.append((surHMM, 'SURPRISE'))

emotions = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD", "SUR"]
emotion_labels = {
    'NEU':'NEUTRAL',
    'ANG':'ANGRY',
    'DIS':'DISGUST',
    'FEA':'FEAR',
    'HAP':'HAPPY',
    'SAD':'SAD',
    'SUR':'SURPRISE'
}

data_summary = pd.read_csv('raw_data_Complete_summary.csv')
data_summary.rename(index={0: data_summary.iat[0,0], 1: data_summary.iat[1,0], 2: data_summary.iat[2,0], 3: data_summary.iat[3,0], 4: data_summary.iat[4,0], 5: data_summary.iat[5,0], 6: data_summary.iat[6,0], 7: data_summary.iat[7,0]}, inplace=True)
data_summary = data_summary.drop(["Unnamed: 0"], axis=1)
data_summary = data_summary.drop(["35"], axis=1)
image_urls = {
    'ANGRY':"https://toppng.com/uploads/preview/emoji-anger-emoticon-iphone-angry-emoji-115630146799aeoop7kx6.png",
    'DISGUST':"https://cdn.shopify.com/s/files/1/1061/1924/products/Poisoned_Emoji_Icon_885fdba4-bbff-40e9-8460-1f453970cbdb.png?v=1485573481",
    'FEAR':"https://p.kindpng.com/picc/s/4-48118_scared-emoji-png-transparent-png.png",
    'HAPPY':"https://toppng.com/uploads/preview/happy-face-emoji-printable-11549848555cni0o8ms6p.png",
    'NEUTRAL':"https://cdn.shopify.com/s/files/1/1061/1924/products/Neutral_Face_Emoji_1024x1024.png?v=1571606037",
    'SAD':"https://cdn.shopify.com/s/files/1/1061/1924/products/Very_sad_emoji_icon_png_large.png?v=1571606089",
    'SURPRISE':"https://www.nicepng.com/png/detail/7-77250_download-surprised-with-teeth-iphone-emoji-icon-in.png"
}

st.set_page_config(page_title="SER System", page_icon=":sound:", layout="wide")

col1, col2, col3 = st.columns([2,10,2])
with col1:
    st.write(' ')
with col2:
    st.title(":smile: :microphone: A Speech Emotion Recongition System using HMM :sound: :smile:")
with col3:
    st.write(' ')
#st.markdown("<h1 style='text-align: center; color: grey;'>:smile: :microphone: A Speech Emotion Recongition System using HMM :sound: :smile:</h1>", unsafe_allow_html=True)
#st.markdown("<h2 style='text-align: center; color: white;'>Upload any wav audio file and see its predicted emotion</h2>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([2,5,1])
with col1:
    st.write(' ')
with col2:
    st.header("Upload any wav audio file and see its predicted emotion")
with col3:
    st.write(' ')

#st.sidebar.subheader("Visualization Settings")

col1, col2, col3 = st.columns(3)
with col1:
    st.write(' ')
with col2:
    uploaded_file = st.file_uploader(label="Upload your wav file", type=['wav'])
with col3:
    st.write(' ')


if uploaded_file is not None:
    try:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(' ')
        with col2:
            st.audio(uploaded_file)
        with col3:
            st.write(' ')
        
        file_name = uploaded_file.name
        contains_emotion = any(label in file_name for label in emotions)
        if contains_emotion:
            for i in emotions:
                res = re.findall(i, file_name)
                if res:
                    break
            #res = [x for x in emotions if re.search(file_name, x)]
            f_label = res[0]
            file_label = emotion_labels[f_label]
        else:
            file_label = "Unknown"
        win, step = 0.03, 0.015
        s, fs = lb.load(uploaded_file, sr=16000)
        reduced_audio = noise_reduction_nonstationary(s, fs)
        f, fn = aSF.feature_extraction(reduced_audio, fs, int(fs*win), int(fs*step))
        f_cut = f[:34]
        features = np.mean(f_cut, axis=1)
        features = features.reshape(-1,1)
        features = features.T
        column_names = list(range(1,36+1))
        column_names = [str(x) for x in column_names]
        df = pd.DataFrame(features)
        
        st.write("The extracted normalized features from the uploaded file:")
        norm_f = get_normalized_features(df)
        #st.write(norm_f)
        
        max_score = -9999999999999999999
        output_label = None
        
        for item in hmm_models:
            hmm_model, label = item
            score = hmm_model.get_score(norm_f)
            if score > max_score:
                max_score = score
                output_label = label

        col1, col2, col3 = st.columns([5,1,5])
        with col1:
            st.write(' ')
        with col2:
            st.write("Labelled Emotion: ",file_label)
            st.write("Predicted Emotion: ",output_label)
        with col3:
            st.write(' ')

        col1, col2, col3 = st.columns([4.2,5,1])
        with col1:
            st.write(' ')
        with col2:
            st.image(image_urls[output_label], width=300)
        with col3:
            st.write(' ')

        st.header("File Graphs")
        st.subheader("Waveform plot")
        fig, ax = plt.subplots(figsize=(20,4))
        ax.plot(s)
        ax.plot(reduced_audio, alpha = 1)
        st.pyplot(fig)

        st.subheader("Zero-crossing rate")
        fig, ax = plt.subplots(figsize=(20,4))
        ax.plot(f_cut[0])
        st.pyplot(fig)

        st.subheader("Energy")
        fig, ax = plt.subplots(figsize=(20,4))
        ax.plot(f_cut[1])
        st.pyplot(fig)

        st.subheader("Entropy of Energy")
        fig, ax = plt.subplots(figsize=(20,4))
        ax.plot(f_cut[2])
        st.pyplot(fig)

        st.subheader("Spectral Centroid")
        fig, ax = plt.subplots(figsize=(20,4))
        ax.plot(f_cut[3])
        st.pyplot(fig)

        st.subheader("Spectral Spread")
        fig, ax = plt.subplots(figsize=(20,4))
        ax.plot(f_cut[4])
        st.pyplot(fig)

        st.subheader("Spectral Entropy")
        fig, ax = plt.subplots(figsize=(20,4))
        ax.plot(f_cut[5])
        st.pyplot(fig)

        st.subheader("Spectral Flux")
        fig, ax = plt.subplots(figsize=(20,4))
        ax.plot(f_cut[6])
        st.pyplot(fig)

        st.subheader("Spectral Rolloff")
        fig, ax = plt.subplots(figsize=(20,4))
        ax.plot(f_cut[7])
        st.pyplot(fig)

        st.subheader("MFCC")
        mfccs = lb.feature.mfcc(y=s, sr=fs, n_mfcc=13)
        fig, ax = plt.subplots(1, figsize=(12,5))
        mfcc_image = librosa.display.specshow(mfccs, sr=fs, x_axis='time')
        col1, col2, col3 = st.columns([1,5,1])
        with col1:
            st.write(' ')
        with col2:
            st.pyplot(fig)
        with col3:
            st.write(' ')

        st.subheader("Chroma")
        fig, ax = plt.subplots(1, figsize=(12,5))
        D = np.abs(lb.stft(s))
        c = lb.feature.chroma_stft(S=D, sr=fs)
        chroma_img = librosa.display.specshow(c, y_axis='chroma', x_axis='time')
        col1, col2, col3 = st.columns([1,5,1])
        with col1:
            st.write(' ')
        with col2:
            st.pyplot(fig)
        with col3:
            st.write(' ')
        
    except Exception as e:
        print(e)
