#imports
import ffmpeg, numpy as np, matplotlib.pyplot as plt
import warnings
import scipy.optimize
import librosa
from scipy import signal, misc
warnings.simplefilter("ignore", DeprecationWarning)
from ffmpeg import Error as FFmpegError

#read in audio from file but add eq
def readAudioWithEQ(filename, params):
    try:
        input_audio, err = (ffmpeg
                    .input(filename)
                    .filter("equalizer", f=abs(params[1]),width_type='q',width=2,gain=params[0])
                    .filter("equalizer", f=abs(params[3]), width_type='q', width=2, gain=params[2])
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

def compute_distance(audio_1, audio_2):
    f_1, t_1, spec_1 = compute_spec(audio_1)
    f_2, t_2, spec_2 = compute_spec(audio_2)
    return np.linalg.norm(spec_1[:-1] - spec_2[:-1])

def compute_distance_mel(audio_1, audio_2):
    spec_1 = librosa.feature.melspectrogram(audio_1, sr=48000)
    spec_2 = librosa.feature.melspectrogram(audio_2, sr=48000)
    distance = np.linalg.norm(spec_1 - spec_2)
    return distance

def loss(params, target_audio, fn_clean, nsamples):
    new_audio = readAudioWithEQ(fn_clean, params)
    new_audio = new_audio[0:nsamples]
    l = compute_distance(new_audio, target_audio)
    return l

####################################################################
####################################################################
####################################################################

# Define target audio
g_true = -10
f_true = 8000
g2_true = 5
f2_true = 2000
target_audio = readAudioWithEQ("test1.wav", np.array( [g_true, f_true, g2_true, f2_true] ))
target_audio = target_audio[0:2*48000]

# lets check that the spetrogram is as expected
f_test, t_test, spec_test = compute_spec(target_audio)

# TODO plot a bette rspectrogram with f and t labels, log scaling of f
plt.imshow(np.log(spec_test))
plt.colorbar()
plt.show()

# sanity check of the loss function (manual)
g_init = -10
f_init = 7900
g2_init = 5
f2_init = 1900

tmp = loss(np.array([g_init, f_init, g2_init, f2_init]), target_audio, "test1.wav", 2*48000)
print(tmp)
tmp2 = loss(np.array([g_true, f_true, g2_true, f2_true]), target_audio, "test1.wav", 2*48000)
print(tmp2)

# ... test the full range (only for g, with fixed f); i.e. a slice of the loss function
gs_test = np.arange(-20, 30)
#test_fs = np.arange(100,10000)
fs_test = 8000
fs_test2 = 2000
test_loss = np.zeros((gs_test.shape[0],))
test_loss2 = np.zeros((gs_test.shape[0],))

for i in range(0,gs_test.shape[0]):
    test_loss[i] = loss(np.array([gs_test[i], 8000, 5, 2000]), target_audio, "test1.wav", 2*48000)
    test_loss2[i] = loss(np.array([-10, 8000, gs_test[i], 2000]), target_audio, "test1.wav", 2*48000)

plt.plot(gs_test,test_loss,'kx-')
plt.plot(gs_test,test_loss2,'kx-')
plt.title("f_test=%s\nf_true=%s" % (fs_test,f_true))
plt.xlabel("g")
plt.ylabel("loss")
plt.show()

### ok run the optimiser on two d

bnds = ((-900, 900), (0, 999999), (-900, 900), (0, 999999))

results = []

for i in range(5):
    #inits
    params = [np.random.randint(-20,20), np.random.randint(0,24000), np.random.randint(-20,20), np.random.randint(0,24000)]
    result = scipy.optimize.minimize(loss, params,
                                    args=(target_audio, "test1.wav",2*48000),
                                    method='SLSQP',
                                    bounds=bnds,
                                    options={'disp':True})
    results += [result]

for result in results:
    print("solution: %s" % result.x)
    print("value: %s" % result.fun)

print('done')