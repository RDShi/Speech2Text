import numpy as np
import os
import re
import json
from scipy.io import wavfile
from python_speech_features import mfcc as get_mfcc
from python_speech_features import delta

data_dir = '/data/srd/data/VCTK-Corpus/'
txt = 'txt'
wav = 'wav48'

wav_max_len = 0
label_max_len = 0

data_list = []
vocabulary = set()
i = 0
for dir_name in os.listdir(os.path.join(data_dir,wav))[:100]:
    for file_name in os.listdir(os.path.join(data_dir,wav,dir_name)):
        sample_name = os.path.join(dir_name,file_name)[:-4]
        if not os.path.exists(os.path.join(data_dir, txt, sample_name)+'.txt'):
            continue
        data_list.append(sample_name)
        i += 1
        if i%1000 == 0:
            print(i)
        with open(os.path.join(data_dir, txt, sample_name)+'.txt', 'r') as fid:
            words = fid.readline().split()
        label_len = 0
        for word in words:
            word = re.sub("[\s\.\!\/_,$%^*()+\"\'?]", "",word)
            if word:
                vocabulary.add(word.lower())
                label_len += 1
                
        label_max_len= max(label_max_len, label_len)
        
        fs, signal = wavfile.read(os.path.join(data_dir, wav, sample_name)+'.wav')
        mfcc = get_mfcc(signal=signal, samplerate=fs, nfft=int(fs*0.025), appendEnergy=True, winfunc=np.hamming)
        wav_len = len(mfcc)
        #wav_len = int(len(signal)/(fs*0.01))
        if wav_len > wav_max_len:
            wav_max_len = wav_len

vocabulary = list(vocabulary)
words_size = len(vocabulary)
print('number of sentence: {}'.format(len(data_list))) #40221
print('number of words in the vocabulary: {}'.format(words_size)) #5706
print('The number of words in the longest sentence: {}'.format(label_max_len)) #37
print('The longest voice length: {}'.format(wav_max_len)) #1927

with open('data/vocabulary.json','w') as fout:
    json.dump(vocabulary, fout, encoding='utf-8')

with open('data/data_list.json','w') as fout:
    json.dump(data_list, fout, encoding='utf-8')


