import numpy as np
import librosa
import scipy

def spec_builder(wav):
    D_1 = librosa.amplitude_to_db(librosa.stft(wav[:66000],n_fft=700, hop_length=258),
                                    amin=0.01,ref=np.mean)
    mel_spec = librosa.feature.melspectrogram(S=D_1,power=1,sr=16000,n_fft=256)
    return mel_spec

def spec_builder_small(wav,power=1.0):
    mel_spec = librosa.feature.melspectrogram(wav[:66000], n_fft=256, hop_length=512,
                                          n_mels=64, sr=22050, power=power)
    mel_spec = librosa.amplitude_to_db(mel_spec, ref=np.max)
    return mel_spec

def make_saver(spec_builder, save_to_file=True):
    def saver(wav, path=None, name=None, spec_builder=spec_builder):
        if save_to_file:
            return spec_builder(wav).tofile(f'{path}/{name}')
        else:
            return spec_builder(wav)
    return saver

def make_transformer(saver, path):
    def trasformer(data_pathes,path=path, length=66000, sr=16000, right_dist=-4, left_dist=-9, func=saver):
        _ = []
        for i in [data_pathes]:
            sr, x = scipy.io.wavfile.read(i)
            if len(x) < length:
                x = np.concatenate([x] * int(np.ceil(length/len(x))))[:length]
                x = x / np.max(np.abs(x))
                spec = saver(x, path, i[left_dist:right_dist])
                spec -= spec.min()
                _.append(spec)
                
            elif len(x) < length + sr:
                x = x / np.max(np.abs(x))
                spec = saver(x[:length], path, i[left_dist:right_dist])
                spec -= spec.min()
                _.append(spec)

            else:
                for j in range((len(x)-length)//sr):
                    feature = x[j*sr : j*sr+length].copy()
                    feature = feature / np.max(np.abs(feature))
                    spec = saver(feature, path, str(i[left_dist:right_dist]+'_'+str(j)))
                    spec -= spec.min()
                    _.append(spec)
        return _
           
    return trasformer
