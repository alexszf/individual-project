#imports
import ffmpeg, numpy as np, matplotlib.pyplot as plt
import warnings
import scipy.optimize
import librosa
from scipy import signal, misc
from skopt import gp_minimize
from skopt.plots import plot_convergence
from ffmpeg import Error as FFmpegError
warnings.simplefilter("ignore", DeprecationWarning)


#read in audio from file
def readAudio(filename):
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

#read in audio from file but add eq
def readAudioWithPipeline(filename, params):
    try:
        input_audio, err = (ffmpeg
                    .input(filename)
                    .filter("volume", volume=params[4])
                    .filter("acompressor", ratio=params[0], threshold=params[1])
                    .filter("equalizer", f=params[2], width_type='q', width=2, gain=params[3])
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

def take_snippet(audio, audio2, start):
    return audio[start:start+(48000*2)], audio2[start:start+(48000*2)]

def loss(x):
    f_source = "snippet_original.wav"
    f_rerecording = "snippet_homemonfacings10.wav"
    target_audio = readAudio(f_source) # read first audio file in clean
    target_audio = target_audio[0:2*48000]
    new_audio = readAudioWithPipeline(f_rerecording, x) # read in the second file, adding the pipeline
    new_audio = new_audio[0:2*48000]
    l = compute_distance(new_audio, target_audio)
    return l

audio1 = readAudio("snippet_original.wav")
audio2 = readAudio("snippet_homemonfacings10.wav")
fft = compute_distance(audio1, audio2)
mfcc = compute_distance_mfcc(audio1, audio2)
print("fft:", fft)
print("mfcc:", mfcc)

bounds = ((1.0, 20.0),          #0 comp ratio
          (0.000976563, 1.0),   #1 comp threshold
          (2.0, 23998.0),       #2 eq central frequency
          (-20.0, 20.0),        #3 eq gain
          (0.0, 20.0))          #4 volume

results = []
for i in range(5):
    result = gp_minimize(loss, bounds,
                     acq_func='EI',
                     n_calls=100,
                     n_random_starts=5,
                     n_points=10000,
                     noise=0.01,
                     verbose=True)
    results += [result]

for i in range(5):
    print("result", (i+1))
    print("location of minimum: %s" % results[i].x)
    print("value of minimum: %s" % results[i].fun)
    plot_convergence(results[i])
    plt.show()