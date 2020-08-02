from pathlib import Path
import librosa
from utils.audio import Audio
import tqdm
import numpy as np
import tensorflow as tf
from random import Random

def mel_wav_from_metafile(wav_folder: str, config: dict, metafile: str, target_dir:str, columns_sep:str='|'):
    audio = Audio(config)
    audio_file_list = []
    wav_folder = Path(wav_folder)
    target_dir = Path(target_dir)
    with open(metafile, 'r', encoding='utf-8') as f:
        for l in f.readlines():
            filename = l.split(columns_sep)[0]
            if filename.endswith('.wav'):
                filename = filename.split('.')[0]
            audio_file_list.append(filename)
    
    for i in tqdm.tqdm(range(len(audio_file_list))):
        filename = audio_file_list[i]
        wav_path = (wav_folder / filename).with_suffix('.wav')
        y, sr = librosa.load(wav_path, sr=audio.config['sampling_rate'])
        mel = audio.mel_spectrogram(y)
        file_path = (target_dir / filename).with_suffix('.npy')
        np.save(file_path, (y, mel))


class Dataset:
    """ Model digestible dataset. """
    
    def __init__(self,
                 file_names: list,
                 preprocessor,
                 batch_size: int,
                 padded_shapes: list,
                 output_types : list,
                 shuffle=True,
                 drop_remainder=True,
                 seed=42):
        self._random = Random(seed)
        self._samples = file_names[:]
        self.preprocessor = preprocessor
        dataset = tf.data.Dataset.from_generator(lambda: self._datagen(shuffle),
                                                 output_types=output_types)
        dataset = dataset.padded_batch(batch_size,
                                       padded_shapes=padded_shapes,
                                       drop_remainder=drop_remainder)
        self.dataset = dataset
        self.data_iter = iter(dataset.repeat(-1))
    
    def next_batch(self):
        return next(self.data_iter)
    
    def all_batches(self):
        return iter(self.dataset)
    
    def _datagen(self, shuffle):
        """
        Shuffle once before generating to avoid buffering
        """
        samples = self._samples[:]
        if shuffle:
            # print(f'shuffling files')
            self._random.shuffle(samples)
        return (self.preprocessor(s) for s in samples)