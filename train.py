import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
from model import convolution_module, encoder, ff_module, mhsa_module
import python_speech_features


if __name__ == "__main__":
    conformer = encoder.ConformerEncoder().cuda()
    
    print("Starting the training of the conformer")

    # loading of the file
    wav, sr = torchaudio.load('./ATCOSIM/WAVdata/gf1/gf1_01/gf1_01_001.wav')
    
    # saving plot to file
    plt.plot(wav[0])
    plt.savefig('./temp/temp.png')

    # extract 80-channel filterbank features from 25ms window with stride of 10ms
    frame_len=0.025 #ms
    frame_shift=0.01 #ms
    wav_feature, energy = python_speech_features.fbank(wav, sr, nfilt=80, winlen=frame_len, winstep=frame_shift)
    

    def normalize_frames(m,epsilon=1e-12):
        return np.array([(v - np.mean(v)) / max(np.std(v),epsilon) for v in m])

    wav_feature = normalize_frames(wav_feature)
    print(wav_feature.shape)

    plt.imshow(wav_feature.T)
    plt.savefig('./temp/fbank.png')

    the_out = conformer(torch.tensor(wav_feature).float().unsqueeze(0).unsqueeze(0).cuda())
    print(the_out.shape) # 1, 19, 256

