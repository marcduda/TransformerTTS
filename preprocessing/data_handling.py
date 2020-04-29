import os
from random import Random

import numpy as np
import tensorflow as tf


class Dataset:
    """ Model digestible dataset. """
    
    def __init__(self,
                 samples,
                 preprocessor,
                 batch_size,
                 shuffle=True,
                 seed=42):
        self._random = Random(seed)
        self._samples = samples[:]
        self.preprocessor = preprocessor
        output_types = (tf.float32, tf.int32, tf.int32)
        padded_shapes = ([-1, preprocessor.mel_channels], [-1], [-1])
        dataset = tf.data.Dataset.from_generator(lambda: self._datagen(shuffle, include_text=False),
                                                 output_types=output_types)
        dataset = dataset.padded_batch(batch_size,
                                       padded_shapes=padded_shapes,
                                       drop_remainder=True)
        self.dataset = dataset
        self.data_iter = iter(dataset.repeat(-1))
    
    def next_batch(self):
        return next(self.data_iter)
    
    def all_batches(self):
        return iter(self.dataset)
    
    def _datagen(self, shuffle, include_text):
        """
        Shuffle once before generating to avoid buffering
        """
        samples = self._samples[:]
        if shuffle:
            # print(f'shuffling files')
            self._random.shuffle(samples)
        return (self.preprocessor(s, include_text=include_text) for s in samples)


def load_files(metafile,
               meldir,
               num_samples=None):
    samples = []
    count = 0
    alphabet = set()
    with open(metafile, 'r', encoding='utf-8') as f:
        for l in f.readlines():
            l_split = l.split('|')
            mel_file = os.path.join(str(meldir), l_split[0] + '.npy')
            text = l_split[1].strip().lower()
            phonemes = l_split[2].strip()
            samples.append((phonemes, text, mel_file))
            alphabet.update(list(text))
            count += 1
            if num_samples is not None and count > num_samples:
                break
        alphabet = sorted(list(alphabet))
        return samples, alphabet


class Tokenizer:
    
    def __init__(self, alphabet, start_token='>', end_token='<', pad_token='/'):
        self.alphabet = alphabet
        self.idx_to_token = {i: s for i, s in enumerate(self.alphabet, start=1)}
        self.idx_to_token[0] = pad_token
        self.token_to_idx = {s: i for i, s in self.idx_to_token.items()}
        self.start_token_index = len(self.alphabet) + 1
        self.end_token_index = len(self.alphabet) + 2
        self.vocab_size = len(self.alphabet) + 3
        self.idx_to_token[self.start_token_index] = start_token
        self.idx_to_token[self.end_token_index] = end_token
    
    def encode(self, sentence, add_start_end=True):
        sequence = [self.token_to_idx[c] for c in sentence if c in self.token_to_idx]
        if add_start_end:
            sequence = [self.start_token_index] + sequence + [self.end_token_index]
        return sequence
    
    def decode(self, sequence):
        return ''.join([self.idx_to_token[int(t)] for t in sequence if int(t) in self.idx_to_token])


class DataPrepper:
    
    def __init__(self,
                 config,
                 tokenizer: Tokenizer,
                 divisible_by: int, ):
        self.start_vec = np.ones((1, config['mel_channels'])) * config['mel_start_value']
        self.end_vec = np.ones((1, config['mel_channels'])) * config['mel_end_value']
        self.tokenizer = tokenizer
        self.mel_channels = config['mel_channels']
        self.divisible_by = divisible_by
    
    def __call__(self, sample, include_text=True):
        phonemes, text, mel_path = sample
        mel = np.load(mel_path)
        return self._run(phonemes, text, mel, include_text=include_text)
    
    def _run(self, phonemes, text, mel, *, include_text):
        encoded_phonemes = self.tokenizer.encode(phonemes)
        extra_end = (self.divisible_by - (mel.shape[-2]+2 % self.divisible_by)) % self.divisible_by
        divisibility_pads = np.zeros(self.end_vec.shape)
        padded_mel = np.concatenate([self.start_vec, mel, self.end_vec, np.tile(divisibility_pads, (extra_end, 1))],
                                  axis=0)
        stop_probs = np.ones((padded_mel.shape[0]))
        stop_probs[len(mel):] = 2
        if include_text:
            return padded_mel, encoded_phonemes, stop_probs, text
        else:
            return padded_mel, encoded_phonemes, stop_probs
