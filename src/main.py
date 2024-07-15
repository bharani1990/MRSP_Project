import os
from matplotlib import pyplot as plt
import pandas as pd
import torch
import numpy as np
from scipy.io import wavfile
import scipy.io.wavfile as wav
from audio_processing import quantize, simulate_aac
from perceptual_loss import percloss
from visualization import plot_audio_comparison, plot_bar


def get_files(path_for_songs):
    dir_list = os.listdir(path_for_songs)
    files = {}
    for entry in dir_list:
        full_path = os.path.join(path_for_songs, entry)
        if os.path.isdir(full_path):
            files[entry] = os.listdir(full_path)
        else:
            files[entry] = entry

    df = pd.DataFrame([(file, genre) for genre, files in files.items() for file in files], columns=['file_name', 'genre'])
    df.reset_index(drop=True, inplace=True)
    df.index += 1
    return df

def main(path_for_songs):
    df = get_files(path_for_songs)
    for i in range(1, len(df) + 1):
        file_name = df.loc[i, 'file_name']
        genre = df.loc[i, 'genre']
        print(file_name, genre)
        file_path = os.path.join(path_for_songs, genre, file_name)
        fs, snd = wavfile.read(file_path)
        if len(snd.shape) > 1:
            snd = np.mean(snd, axis=1)
        snd = snd.astype(float) / np.max(np.abs(snd))
        snd_quant = quantize(snd, 4)
        snd_aac = simulate_aac(snd)

        # Save the quantized and aac files in the results folder
        wav.write('results/quantized_'+file_name, fs, (snd_quant * 32767).astype(np.int16))
        wav.write('results/aac_'+file_name, fs, (snd_aac * 32767).astype(np.int16))

        if len(snd.shape) == 2 or len(snd_quant.shape) == 2 or len(snd_aac.shape) == 2:
            snd = snd[:, 0]
            snd_quant = snd_quant[:, 0]
        else:
            snd = snd
            snd_quant = snd_quant
        ploss_quant = percloss(torch.from_numpy(snd).float(), torch.from_numpy(snd_quant).float(), fs)
        ploss_aac = percloss(torch.from_numpy(snd).float(), torch.from_numpy(snd_aac).float(), fs)
        print(ploss_quant, ploss_aac, ploss_aac/ploss_quant)
        # plot_audio_comparison(snd, snd_quant, snd_aac, fs)
        df.loc[i, 'ploss_Q'] = ploss_quant.item()
        df.loc[i, 'ploss_A'] = ploss_aac.item()
    df['ploss_A/Q'] = df['ploss_A'] / df['ploss_Q']
    df['winner'] = np.where(df['ploss_A/Q'] < 1, 'AAC', 'Quantization')
    df.to_csv('results/ploss_AQ.csv', index=False)
    plot_bar(df)
    
    

if __name__ == '__main__':
    main('data/')
