import random
import numpy as np
import os
import re
from scipy.io import wavfile
from python_speech_features import mfcc as get_mfcc
from python_speech_features import delta


def get_batch(batch_size, vocabulary, data_list, wav_max_len=1927, label_max_len=37, data_dir = '/data/srd/data/VCTK-Corpus/', txt = 'txt', wav = 'wav48'):
    
    batch_feats = []
    batch_labs = []

    for i in range(batch_size):
        sample_name = random.choice(data_list)
        feat, label = get_sample(sample_name,vocabulary, wav_max_len, label_max_len, data_dir, txt, wav)
        
        batch_feats.append(feat)
        batch_labs.append(label)

    return batch_feats, batch_labs


def get_sample(sample_name,vocabulary, wav_max_len=1927, label_max_len=37, data_dir = '/data/srd/data/VCTK-Corpus/', txt = 'txt', wav = 'wav48'):
    
    fs, signal = wavfile.read(os.path.join(data_dir, wav, sample_name)+'.wav')
    mfcc = get_mfcc(signal=signal, samplerate=fs, nfft=int(fs*0.025), appendEnergy=True, winfunc=np.hamming)
    mfcc_1 = delta(mfcc, 1)
    mfcc_2 = delta(mfcc_1, 1)
    feat = np.hstack([mfcc, mfcc_1, mfcc_2])
    
    l, n = feat.shape
    zero_padding = np.zeros([wav_max_len-l ,n])
    feat = np.concatenate((feat,zero_padding))
    
    with open(os.path.join(data_dir, txt, sample_name)+'.txt', 'r') as fid:
        words = fid.readline().split()
    
    label = []
    for word in words:
        word = re.sub("[\s\.\!\/_,$%^*()+\"\'?]", "",word)
        if word:
            label.append(vocabulary.index(word.lower())+1)

    while len(label)<label_max_len:
        label.append(0)
 
    return feat, np.array(label).astype(np.int32)


def format_time(time):
    """ It formats a datetime to print it
        Args:
            time: datetime
        Returns:
            a formatted string representing time
    """
    m, s = divmod(time, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    return ('{:02d}d {:02d}h {:02d}m {:02d}s').format(int(d), int(h), int(m), int(s))
