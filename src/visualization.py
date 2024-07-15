import matplotlib.pyplot as plt
import librosa.display
import numpy as np

def plot_audio_comparison(snd, snd_quant, snd_aac, fs):
    plt.figure(figsize=(15, 5))

    snd_float = snd.astype(np.float32) / np.max(np.abs(snd))
    snd_quant_float = snd_quant.astype(np.float32) / np.max(np.abs(snd_quant))
    snd_aac_float = snd_aac.astype(np.float32) / np.max(np.abs(snd_aac))

    plt.subplot(1, 3, 1)
    librosa.display.waveshow(snd_float, sr=fs)
    plt.title('Original Audio')

    plt.subplot(1, 3, 2)
    librosa.display.waveshow(snd_quant_float, sr=fs)
    plt.title('Quantized Audio')

    plt.subplot(1, 3, 3)
    librosa.display.waveshow(snd_aac_float, sr=fs)
    plt.title('AAC Encoded Audio')

    plt.tight_layout()
    plt.show()

def plot_bar(df):
    new_df = df.copy()
    new_df['ploss_Q_norm'] = new_df.apply(lambda row: row['ploss_Q'] / max(row['ploss_Q'], row['ploss_A']), axis=1)
    new_df['ploss_A_norm'] = new_df.apply(lambda row: row['ploss_A'] / max(row['ploss_Q'], row['ploss_A']), axis=1)
    fig, ax = plt.subplots(figsize=(15, 8))
    bar_width = 0.4
    index = np.arange(len(new_df))
    bars1 = ax.bar(index, new_df['ploss_Q_norm'], bar_width, label='Quantized Loss')
    bars2 = ax.bar(index + bar_width, new_df['ploss_A_norm'], bar_width, label='AAC Loss')
    ax.set_xlabel('File Name')
    ax.set_ylabel('Normalized Perceptual Loss')
    ax.set_title('Comparison of Quantized and AAC Loss for Different Audio Files')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(new_df['file_name'], rotation=90)
    ax.legend()
    fig.tight_layout()
    plt.savefig('results/ploss_AQ.png')
    plt.show()
