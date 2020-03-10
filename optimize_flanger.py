#imports
import ffmpeg, numpy as np, matplotlib.pyplot as plt
import warnings
import scipy.optimize
import librosa
from scipy import signal, misc
warnings.simplefilter("ignore", DeprecationWarning)
from ffmpeg import Error as FFmpegError

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
def readAudioWithEQ(filename, params):
    try:
        input_audio, err = (ffmpeg
                    .input(filename)
                    .filter("flanger", speed=params[0])
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

def compute_distance_mel(audio_1, audio_2):
    spec_1 = librosa.feature.melspectrogram(audio_1, sr=48000)
    spec_2 = librosa.feature.melspectrogram(audio_2, sr=48000)
    distance = np.linalg.norm(spec_1 - spec_2)
    return distance

def compute_distance(audio_1, audio_2):
    f_1, t_1, spec_1 = compute_spec(audio_1)
    f_2, t_2, spec_2 = compute_spec(audio_2)
    return np.linalg.norm(spec_1[:-1] - spec_2[:-1])

def loss(params, target_audio, fn_clean, nsamples):
    new_audio = readAudioWithEQ(fn_clean, params)
    new_audio = new_audio[0:nsamples]

    l = compute_distance(new_audio, target_audio)
    return l

####################################################################
####################################################################
####################################################################

# Define target 

fla_s_true = 8
target_audio = readAudioWithEQ("test1.wav", np.array( [fla_s_true] ))
target_audio = target_audio[0:2*48000]

# lets check that the spetrogram is as expected
f_test, t_test, spec_test = compute_spec(target_audio)

# TODO plot a better spectrogram with f and t labels, log scaling of f
plt.imshow(np.log(spec_test))
plt.colorbar()
plt.show()

# sanity check of the loss function (manual)
fla_s_init = 7.95

print("initial loss at first guess")
tmp = loss(np.array([fla_s_init]), target_audio, "test1.wav", 2*48000)
print(tmp)
print("loss at correct guess")
tmp2 = loss(np.array([fla_s_true]), target_audio, "test1.wav", 2*48000)
print(tmp2)

# ... test the full range of speeds; i.e. a slice of the loss function
speed_test = np.arange(0.1, 10, 0.1)
#test_fs = np.arange(100,10000)
test_loss = np.zeros((speed_test.shape[0],))

for i in range(0,speed_test.shape[0]):
    test_loss[i] = loss(np.array([speed_test[i]]), target_audio, "test1.wav", 2*48000)

plt.plot(speed_test, test_loss,'kx-')
plt.title("testing for true value of 8")
plt.xlabel("speed")
plt.ylabel("loss")
plt.show()

bnds = ((0.1, 10),)

### ok run the optimiser on two d
result = scipy.optimize.minimize(loss, [fla_s_init],
                                 args=(target_audio, "test1.wav",2*48000),
                                 method='TNC',
                                 bounds=bnds,
                                 options={'disp':True})

print("solution: %s" % result.x)
print('done')
