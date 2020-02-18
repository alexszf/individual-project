import ffmpeg, numpy as np
import matplotlib.pyplot as plt
import warnings
import scipy.signal, scipy.optimize
import librosa
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
def readAudioWithEQ(filename):
    try:
        input_audio, err = (ffmpeg
                    .input(filename)
                    .filter("equalizer", f=1000, t='q', w=100, g=10)
                    .output('-', format='s16le', acodec='pcm_s16le', ac=1, ar='48k')
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True)
                    )
    except ffmpeg.Error as e:
        print(e.stderr)
    read_audio = np.fromstring(input_audio, dtype=np.int16).astype(np.float16)
    return read_audio

def compute_distance(audio_1, audio_2):
    if (np.shape(audio_1) != np.shape(audio_2)):
        print("audio files are not the same shape")
    f_1, t_1, spec_1 = scipy.signal.spectrogram(audio_1, fs=48000)
    f_2, t_2, spec_2 = scipy.signal.spectrogram(audio_2, fs=48000)
    return np.linalg.norm(spec_1[:-1] - spec_2[:-1])

#incomplete attempt to use librosa for spectrograms
def compute_distance_librosa(audio_1, audio_2):
    spec_1 = librosa.feature.melspectrogram(audio_1, sr=48000)
    spec_2 = librosa.feature.melspectrogram(audio_2, sr=48000)
    return np.linalg.norm(spec_1 - spec_2)

def apply_effect(params, clean_audio):
    r_gain = 10
    r_freq = params[0]
    r_width = 100
    #ffmpeg fails if eq filter extends below 0Hz
    if (r_freq < r_width):
        r_freq = r_width
    x = np.round(clean_audio).astype(np.int16)
    new_audio = None
    try:
        #define graph
        process_audio = (ffmpeg
            .input('pipe:', format='s16le', acodec='pcm_s16le', ac=1, ar='48k')
            .filter("equalizer", f=r_freq, t='q', w=r_width, g=r_gain)
            .output('pipe:', format='s16le', acodec='pcm_s16le', ac=1, ar='48k')
            .run_async(pipe_stdin=True, pipe_stdout=True)
        )
        #pipe in the audio
        process_audio.stdin.write(
            x
            .astype(np.int16)
            .tobytes()
        )
        process_audio.stdin.close()
        signal = process_audio.stdout.read()
        new_audio = np.fromstring(signal, dtype=np.int16).astype(np.float16)
        process_audio.stdout.close()
    except ffmpeg.Error as e:
        #print(e.stderr)
        sys.exit(1)
    return new_audio

def func(params, target_audio, clean_audio):
    new_audio = apply_effect(params, clean_audio)
    #return distance
    return compute_distance(new_audio, target_audio)

audio_withEQ = readAudioWithEQ("../recordings/sample_ffmpeg.wav")
audio_clean = readAudio("../recordings/sample_ffmpeg.wav")
audio_withEQ = audio_withEQ[96000:144000]
audio_clean = audio_clean[96000:144000]




#frequencies = np.arange(3000, step=5)
#distances = []
#
#for freq in frequencies:
#    audio_withEQ2 = apply_effect([freq], audio_clean)
#    distances += [compute_distance(audio_withEQ, audio_withEQ2)]
#print(distances)
#
#plt.plot(distances)
#plt.ylim(0, 5000)
#plt.show()