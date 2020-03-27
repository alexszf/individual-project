#imports
import ffmpeg, numpy as np, matplotlib.pyplot as plt
import warnings
import scipy.optimize
import librosa
from scipy import signal, misc
from skopt import gp_minimize
from skopt.plots import plot_convergence
warnings.simplefilter("ignore", DeprecationWarning)
from ffmpeg import Error as FFmpegError

filename_source = "../recordings/dataset/source.wav"
filename_rerecording = "../recordings/dataset/home mon face s10.wav"

def retrieve_audio(filename):
    try:
        input_audio, err = (ffmpeg
                    .input(filename)
                    .output('-', format='s16le', acodec='pcm_s16le', ac=1, ar='48k')
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                    )
    except ffmpeg.Error as e:
        print(e.stderr)
    read_audio = np.fromstring(input_audio, dtype=np.int16).astype(np.float16)
    return read_audio

def compute_spec(audio):
    fbase, tbase, spec = signal.spectrogram(audio, fs=48000)
    return fbase, tbase, spec

def compute_distance_mfcc(audio_1, audio_2):
    spec_1 = librosa.feature.mfcc(audio_1, sr=48000)
    spec_2 = librosa.feature.mfcc(audio_2, sr=48000)
    distance = np.linalg.norm(spec_1 - spec_2)
    return distance

def compute_distance(audio_1, audio_2):
    f_1, t_1, spec_1 = compute_spec(audio_1)
    f_2, t_2, spec_2 = compute_spec(audio_2)
    return np.linalg.norm(spec_1[:-1] - spec_2[:-1])

def loss(x):
    fla_s_true = 8
    target_audio = readAudioWithEQ("test1.wav", np.array([fla_s_true]))
    target_audio = target_audio[0:2*48000]
    new_audio = readAudioWithEQ("test1.wav", x)
    new_audio = new_audio[0:2*48000]
    l = compute_distance(new_audio, target_audio)
    return l

def take_snippet(audio, audio2, start):
    return audio[start:start+(48000*5)], audio2[start:start+(48000*5)]


#choose a random starting sample
#startpoint = np.random.randint(0, 8640000-48000)

#for the purposes of testing
start = 2000000

source, rerecording = take_snippet(retrieve_audio(filename_source), retrieve_audio(filename_rerecording), start)
#take a random corresponding 5 second segment from each the source recording and the rerecording

print("fft", compute_distance(source, rerecording))
print("mfcc", compute_distance_mfcc(source, rerecording))

plt.plot(source, color="green")
plt.plot(rerecording, color="red")
plt.show()