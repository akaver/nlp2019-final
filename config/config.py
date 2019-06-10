from pathlib import Path
import multiprocessing

BASE_DIR = Path('.')

config = {
    'data': {
        'raw_train_data_path': BASE_DIR / 'data/raw/Train.csv',
        'raw_test_file_path': BASE_DIR / 'data/raw/Test.csv',

        # data preprocessor output
        'data_path': BASE_DIR / 'data/processed/',

        'train_file_path': BASE_DIR / 'data/processed/training.csv',
        'test_file_path': BASE_DIR / 'data/processed/test.csv',
        'validation_file_path': BASE_DIR / 'data/processed/validation.csv',
        'label_file_path': BASE_DIR / 'data/processed/labels.csv',

        # how much data to use and ratio for splitting between train/validation sets
        'data_limit_train': 50_000,  # 50_000, # data items to use for train/validation. None for no limit
        'data_limit_test': None,
        'data_split_train': 0.8,  # split data between train and validate

        # how many most frequent labels to use
        'label_limit': 20,  # None to use all found labels
        'eval_size': 0.2,
    },

    'output': {
        'path': BASE_DIR / 'output',
        'log_dir': BASE_DIR / 'output/log',
    },
    'train': {
        'seed': 1,
        'n_gpu': [0],
        'num_workers': multiprocessing.cpu_count(),
        'weight_decay': 1e-5,
        'fp16': False,
        'do_lower_case': True,
        "do_train": True,
        "do_eval": True,
        "train_size": -1,
        "val_size": -1,
        "train_batch_size": 16,
        "eval_batch_size": 16,
        "max_seq_length": 128,
        "gradient_accumulation_steps": 1,
        "num_train_epochs": 4,
        "learning_rate": 3e-5,
        "loss_scale": 128,
        "warmup_proportion": 0.1,
    },
    'bert': {
        'path':  BASE_DIR / 'model/pretrain/uncased_L-12_H-768_A-12/',
        'cache': BASE_DIR / 'model/pretrain/uncased_L-12_H-768_A-12/cache/',

        'vocab_path': BASE_DIR / 'model/pretrain/uncased_L-12_H-768_A-12/vocab.txt',
        'tf_checkpoint_path': BASE_DIR / 'model/pretrain/uncased_L-12_H-768_A-12/bert_model.ckpt',
        'bert_config_file': BASE_DIR / 'model/pretrain/uncased_L-12_H-768_A-12/bert_config.json',

        'bert_model_dir': BASE_DIR / 'model/pretrain/pytorch_pretrain',
        'pytorch_model_path': BASE_DIR / 'model/pretrain/pytorch_pretrain/pytorch_model.bin',
    },
    'model': {
        'arch': 'bert'
    }
}