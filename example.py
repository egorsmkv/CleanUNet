import torch

from scipy.io.wavfile import write as wavwrite

from cleanunet.network import CleanUNet

# load the checkpoint
checkpoint = torch.load('./DNS-large-full.pkl', map_location='cpu')

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
