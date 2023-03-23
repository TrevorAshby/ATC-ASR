import torch
import torchaudio
from model import convolution_module, encoder, ff_module, mhsa_module


if __name__ == "__main__":
    conformer = encoder.ConformerEncoder().cuda()
    
    print("Starting the training of the conformer")
