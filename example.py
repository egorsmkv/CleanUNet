import torch
import torchaudio

from scipy.io.wavfile import write as wavwrite

from cleanunet.network import CleanUNet
from cleanunet.util import sampling

# load the checkpoint
checkpoint = torch.load('./DNS-large-full.pkl', map_location='cpu')

network_config = {
        "channels_input": 1,
        "channels_output": 1,
        "channels_H": 64,
        "max_H": 768,
        "encoder_n_layers": 8,
        "kernel_size": 4,
        "stride": 2,
        "tsfm_n_layers": 5, 
        "tsfm_n_head": 8,
        "tsfm_d_model": 512, 
        "tsfm_d_inner": 2048
}

# build the network
net = CleanUNet(**network_config)
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()

# load the file
noisy_audio, sr = torchaudio.load('./demo/tsn_2.wav')

# clean the noisy speech
generated_audio = sampling(net, noisy_audio)

# get the cleaned speech
cleaned_audio = generated_audio[0].squeeze().cpu().numpy()

# save the cleaned speech
wavwrite('tsn_2_enhanced.wav', 16_000, cleaned_audio)
