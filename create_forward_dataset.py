import tensorflow as tf
import numpy as np
from tqdm import tqdm

from utils.config_manager import Config
from utils.logging import SummaryManager
from preprocessing.datasets import TextMelDurDataset, ForwardPreprocessor
from models.transformer.transformer_utils import create_mel_padding_mask
from utils.scripts_utils import dynamic_memory_allocation, basic_train_parser

np.random.seed(42)
tf.random.set_seed(42)

dynamic_memory_allocation()

parser = basic_train_parser()
parser.add_argument('--forward_weights', type=str, default='')
args = parser.parse_args()
config_manager = Config(config_path=args.config, model_kind='forward')
config_manager.create_remove_dirs(clear_dir=args.clear_dir,
                                  clear_logs=args.clear_logs,
                                  clear_weights=args.clear_weights)
config_manager.print_config()
if args.forward_weights != '':
    model = config_manager.load_model(args.forward_weights)
else:
    model = config_manager.load_model()
# config_manager.compile_model(model)
preproc = ForwardPreprocessor(config_manager.config, tokenizer=model.text_pipeline.tokenizer)
data_handler = TextMelDurDataset.default_all_from_config(config_manager, preprocessor=preproc)

target_dir = config_manager.train_datadir / f'forward_mels'
target_dir.mkdir(exist_ok=True)
config_manager.dump_config()
script_batch_size = config_manager.config['batch_size']
dataset = data_handler.get_dataset(script_batch_size, shuffle=False, drop_remainder=False)
summary_manager = SummaryManager(model=model, log_dir=config_manager.log_dir, config=config_manager.config,
                                 default_writer='ForwardMels')
# checkpoint = tf.train.Checkpoint(step=tf.Variable(1),
#                                  optimizer=model.optimizer,
#                                  net=model)
# manager = tf.train.CheckpointManager(checkpoint, config_manager.weights_dir,
#                                      max_to_keep=config_manager.config['keep_n_weights'],
#                                      keep_checkpoint_every_n_hours=config_manager.config['keep_checkpoint_every_n_hours'])
# if args.forward_weights == '':
#     checkpoint.restore(manager.latest_checkpoint)
#     if manager.latest_checkpoint:
#         print(f'\nresuming training from step {model.step} ({manager.latest_checkpoint})')
#     else:
#         print(f'\nCould NOT load weights. Check config.')

if config_manager.config['debug'] is True:
    print('\nWARNING: DEBUG is set to True. Training in eager mode.')
iterator = tqdm(enumerate(dataset.all_batches()))

for c, (mel_batch, text_batch, durations_batch, file_name_batch) in iterator:
    iterator.set_description(f'Processing dataset')
    model_out = model.val_step(input_sequence=text_batch,
                               target_sequence=mel_batch,
                               target_durations=durations_batch)
    
    pred_mel = model_out['mel'].numpy()
    mask = create_mel_padding_mask(mel_batch)
    pred_mel = tf.expand_dims(1 - tf.squeeze(create_mel_padding_mask(mel_batch)), -1) * pred_mel
    mel = pred_mel.numpy()
    unpad_mask = create_mel_padding_mask(pred_mel)
    for i, name in enumerate(file_name_batch):
        step = c*script_batch_size + 1
        mel_len = int(unpad_mask[i].shape[-1] - np.sum(unpad_mask[i]))
        unpad_mel = mel[i][:mel_len, :]
        file_name = name.numpy().decode('utf-8')
        if i == 0:
            target_mel = mel_batch[i][:mel_len, :]
            summary_manager.display_mel(tag='ForwardDataset/ForwardMels', mel=unpad_mel, step=step)
            summary_manager.display_audio(tag=f'ForwardDataset/InvertedTargetMel', mel=target_mel, step=step)
            summary_manager.display_audio(tag=f'ForwardDataset/InvertedPredictedMel', mel=unpad_mel, step=step)
        np.save(str(target_dir / f'{file_name}.npy'), unpad_mel)

print('Done.')
