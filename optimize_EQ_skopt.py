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

#read in audio from file but add eq
def readAudioWithEQ(filename, params):
    try:
        input_audio, err = (ffmpeg
                    .input(filename)
                    .filter("equalizer", f=params[1], width_type='q', width=2, gain=params[0])
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
    g_true = -10
    f_true = 1000
    target_audio = readAudioWithEQ("test2.wav", np.array([g_true, f_true]))
    target_audio = target_audio[0:2*48000]
    new_audio = readAudioWithEQ("test2.wav", x)
    new_audio = new_audio[0:2*48000]
    l = compute_distance(new_audio, target_audio)
    return l

#frequencies = [2000, 4000, 8000]

#for freq in frequencies:
#    target_audio = readAudioWithEQ("test2.wav", np.array([-10, freq]))
#    #gs_test = np.arange(-30,30, 0.5)
#    fs_test = np.arange(100, 8000, 100)
#    test_loss = np.zeros((fs_test.shape[0],))
#
#    for i in range(0,fs_test.shape[0]):
#        audio = readAudioWithEQ("test2.wav", np.array([-10, fs_test[i]]))
#        test_loss[i] = compute_distance_mfcc(audio, target_audio)

#    plt.plot(fs_test, test_loss,'kx-')
#    plt.title("f_test=%s\nf_true=%s" % (freq,freq))
#    plt.xlabel("g")
#    plt.ylabel("loss")
#    plt.show()

space = ((-20.0, 20.0), (2.0, 23998.0))
results = []
for i in range(5):
    result = gp_minimize(loss, space,
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