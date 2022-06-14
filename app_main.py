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
import eyed3
import pydub

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
    'FEAR':"data:image/jpeg;base64,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",
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
