from pathlib import Path
from random import Random

import tensorflow as tf

from utils.audio import Audio

class MetadataToDataset:
    def __init__(self,
                 data_directory,
                 preprocessor,
                 audio_module: Audio,
                 metadata_filename: str,
                 wav_dirname: str,
                 metadata_reader=None,
                 max_wav_len=None):
        
        self.audio = audio_module
        if metadata_reader is not None:
            self.metadata_reader = metadata_reader
        else:
            self.metadata_reader = self._default_metadata_reader
        self.data_directory = Path(data_directory)
        self.metadata_path = self.data_directory / metadata_filename
        self.wav_directory = self.data_directory / wav_dirname
        self.data = self._build_file_list()
        self.preprocessor = preprocessor
        self.max_wav_len = max_wav_len
    
    def _default_metadata_reader(self, metadata_path, column_sep='|'):
        wav_list = []
        text_list = []
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for l in f.readlines():
                l_split = l.split(column_sep)
                filename, text = l_split[0], l_split[-1]
                if filename.endswith('.wav'):
                    filename = filename.split('.')[0]
                wav_list.append(filename)
                text_list.append(text)
        return wav_list, text_list
    
    def _build_file_list(self):
        wav_list, text_list = self.metadata_reader(self.metadata_path)
        file_list = [x.with_suffix('').name for x in self.wav_directory.iterdir() if x.suffix == '.wav']
        for metadata_item in wav_list:
            assert metadata_item in file_list, f'Missing file: metadata item {metadata_item}, was not found in {self.wav_directory}.'
        return list(zip(wav_list, text_list))
    
    def read_wav(self, wav_path, desired_channels=1, desired_samples=-1, name=None):
        file = tf.io.read_file(wav_path)
        # TODO: missing resampling if sr is different than audio.config['sample_rate']
        y, sr = tf.audio.decode_wav(file,
                                    desired_channels=desired_channels,
                                    desired_samples=desired_samples,
                                    name=name)
        y = tf.squeeze(y)
        if self.max_wav_len is not None:
            y_len = tf.shape(y)[0]
            offset = tf.random.uniform([1], 0, max(1, y_len - self.max_wav_len), dtype=tf.int32)[0]
            y = y[offset: offset + self.max_wav_len]
        return y
    
    def wav_to_mel(self, wav):
        return self.audio.mel_spectrogram_tf(wav)
    
    def _read_sample(self, sample, mel = None):
        wav = self.read_wav((self.wav_directory / sample[0]).with_suffix('.wav').as_posix())
        if mel is None:
            mel = self.wav_to_mel(wav)
        text = sample[1]
        return mel, text, wav
    
    def _process_sample(self, sample):
        mel, text, wav = self._read_sample(sample)
        return self.preprocessor(mel=mel, text=text, wav=wav)
    
    def get_dataset(self, batch_size, shuffle=True, drop_remainder=True):
        # TODO: these should be define with preprocessor
        output_types = (tf.float32, tf.float32)
        padded_shapes = ([-1, self.audio.config['mel_channels']], [-1, 1])
        return Dataset(
            samples=self.data,
            preprocessor=self._process_sample,
            batch_size=batch_size,
            output_types=output_types,
            padded_shapes=padded_shapes,
            shuffle=shuffle,
            drop_remainder=drop_remainder)
    
    @classmethod
    def get_default_training_from_config(cls, config, preprocessor, metadata_reader=None):
        audio = Audio(config)
        return cls(data_directory=config['data_directory'],
                   preprocessor=preprocessor,
                   audio_module=audio,
                   metadata_reader=metadata_reader,
                   metadata_filename=config['train_metadata_filename'],
                   wav_dirname=config['wav_subdir_name'],
                   max_wav_len=config['max_wav_segment_lenght'])
    
    @classmethod
    def get_default_validation_from_config(cls, config, preprocessor, metadata_reader=None, max_wav_len=None):
        audio = Audio(config)
        if max_wav_len is None:
            max_wav_len = config['max_wav_segment_lenght']
        return cls(data_directory=config['data_directory'],
                   preprocessor=preprocessor,
                   audio_module=audio,
                   metadata_reader=metadata_reader,
                   metadata_filename=config['valid_metadata_filename'],
                   wav_dirname=config['wav_subdir_name'],
                   max_wav_len=max_wav_len)


class Dataset:
    """ Model digestible dataset. """
    
    def __init__(self,
                 samples: list,
                 preprocessor,
                 batch_size: int,
                 padded_shapes: tuple,
                 output_types: tuple,
                 shuffle=True,
                 drop_remainder=True,
                 seed=42):
        self._random = Random(seed)
        self._samples = samples[:]
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
            self._random.shuffle(samples)
        return (self.preprocessor(s) for s in samples)


class MelGANPreprocessor:
    def __call__(self, mel, text, wav):
        return mel, tf.expand_dims(wav, -1)

# def mel_wav_from_metafile(wav_folder: str, config: dict, metafile: str, target_dir: str, columns_sep: str = '|'):
#     audio = Audio(config)
#     audio_file_list = []
#     wav_folder = Path(wav_folder)
#     target_dir = Path(target_dir)
#     with open(metafile, 'r', encoding='utf-8') as f:
#         for l in f.readlines():
#             filename = l.split(columns_sep)[0]
#             if filename.endswith('.wav'):
#                 filename = filename.split('.')[0]
#             audio_file_list.append(filename)
#
#     for i in tqdm.tqdm(range(len(audio_file_list))):
#         filename = audio_file_list[i]
#         wav_path = (wav_folder / filename).with_suffix('.wav')
#         y, sr = librosa.load(wav_path, sr=audio.config['sampling_rate'])
#         mel = audio.mel_spectrogram(y)
#         file_path = (target_dir / filename).with_suffix('.npy')
#         np.save(file_path, (y, mel))
