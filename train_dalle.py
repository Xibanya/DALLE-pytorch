import argparse
from pathlib import Path
import time
from glob import glob
import os
from tqdm import tqdm
import shutil

import torch
import wandb
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, Adamax, AdamW, RAdam, NAdam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from dalle_pytorch import OpenAIDiscreteVAE, VQGanVAE, DiscreteVAE, DALLE
from dalle_pytorch import distributed_utils
from dalle_pytorch.loader import TextImageDataset
from dalle_pytorch.tokenizer import tokenizer, HugTokenizer, ChineseTokenizer, YttmTokenizer

# libraries needed for webdataset support
import webdataset as wds
from torchvision import transforms as T
from PIL import Image
from io import BytesIO

CFG_PATH = "../configs/train_config.yaml"
CFG_EXISTS = Path(CFG_PATH).exists()

if CFG_EXISTS:
    with open("../configs/train_config.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

USE_CFG = CFG_EXISTS and cfg['use_config']

RESUME_PATH = cfg['dalle_path'] if USE_CFG else './dalle.pt'
OUTPUT_NAME = cfg['dalle_output_file_name'] if USE_CFG else 'dalle'
WANDB_NAME = cfg['wandb_name'] if USE_CFG else 'dalle_train_transformer'
DATASET_PATH = cfg['image_text_folder'] if USE_CFG else '../Dataset'

ADAM = 'Adam'  # works
ADAMAX = 'Adamax'
ADAMW = 'AdamW'  # works
RADAM = 'RAdam'  # works
NADAM = 'NAdam'
OPTIMIZER = cfg['optimizer'] if USE_CFG else ADAM

LEARNING_RATE = float(cfg['learning_rate']) if USE_CFG else 3e-4
EPOCH_COUNT = cfg['epochs'] if USE_CFG else 5
DEPTH_COUNT = cfg['depth'] if USE_CFG else 8
START_BATCH_SIZE = cfg['batch_size'] if USE_CFG else 4
CLIP_GRAD_NORM = cfg['clip_grad_norm'] if USE_CFG else 0.5
START_HEADS = cfg['heads'] if USE_CFG else 8
IMG_LOSS_WEIGHT = cfg['img_loss_weight'] if USE_CFG else 7
MODEL_DIM_HEAD = cfg['dim_head'] if USE_CFG else 64
RESIZE_RATIO = cfg['resize_ratio'] if USE_CFG else 0.75

A_DROPOUT = cfg['attention_dropout'] if USE_CFG else 0.0
F_DROPOUT = cfg['ff_dropout'] if USE_CFG else 0.0
GEN_IMG_STEPS = cfg['img_gen_steps'] if USE_CFG else 50
TOP_K = cfg['top_k'] if USE_CFG else [0.7, 0.8, 0.9]
TEMPERATURE = cfg['temperature'] if USE_CFG else [1.0, 1.0, 1.0]

VQ_CHECKPOINT = None
VQ_CONFIG = None


class Colors:
    HEADER = '\033[95m'; OKBLUE = '\033[94m'; OKCYAN = '\033[96m'; OKGREEN = '\033[92m'
    WARNING = '\033[93m'; FAIL = '\033[91m'; ENDC = '\033[0m'; BOLD = '\033[1m'; UNDERLINE = '\033[4m'


def print_cyan(msg):
    print(f"{Colors.OKCYAN}{msg}{Colors.ENDC}")


def print_blue(msg):
    print(f"{Colors.OKBLUE}{msg}{Colors.ENDC}")


def print_green(msg):
    print(f"{Colors.OKGREEN}{msg}{Colors.ENDC}")


def print_warn(msg):
    print(f"{Colors.WARNING}{msg}{Colors.ENDC}")


parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=False)

group.add_argument('--vae_path', type=str,
                   #default='./characters_vae.pt',
                   default='./city_vae-final.pt',
                   help='path to your trained discrete VAE')

group.add_argument('--dalle_path', type=str, default=RESUME_PATH,
                   help='path to your partially trained DALL-E')

parser.add_argument('--vqgan_model_path', type=str, default=VQ_CHECKPOINT,
                    help='path to your trained VQGAN weights. This should be a .ckpt file. '
                         '(only valid when taming option is enabled)')

parser.add_argument('--vqgan_config_path', type=str, default=VQ_CONFIG,
                    help='path to your trained VQGAN config. This should be a .yaml file. '
                         '(only valid when taming option is enabled)')

parser.add_argument('--image_text_folder', type=str,
                    default=DATASET_PATH,
                    help='path to your folder of images and text for learning the DALL-E')

parser.add_argument('--wds', type=str, default='',
                    help='Comma separated list of WebDataset (1) image and (2) text column names. '
                         'Must contain 2 values, e.g. img,cap.')

parser.add_argument('--truncate_captions', dest='truncate_captions', action='store_true',
                    help='Captions passed in which exceed the max token length will be '
                         'truncated if this is set.')

parser.add_argument('--random_resize_crop_lower_ratio', dest='resize_ratio', type=float,
                    default=RESIZE_RATIO,
                    help='Random resized crop lower ratio')

parser.add_argument('--chinese', dest='chinese', action='store_true')

parser.add_argument('--taming', dest='taming', action='store_true')

parser.add_argument('--hug', dest='hug', action='store_true')

parser.add_argument('--bpe_path', type=str,
                    help='path to your BPE json file')

parser.add_argument('--dalle_output_file_name', type=str,
                    default=OUTPUT_NAME,
                    help='output_file_name')

parser.add_argument('--fp16', action='store_true',
                    help='(experimental) - Enable DeepSpeed 16 bit precision. Reduces VRAM.')

parser.add_argument('--amp', action='store_true',
                    help='Apex "O1" automatic mixed precision. More stable than 16 bit precision. '
                         'Can\'t be used in conjunction with deepspeed zero stages 1-3.')

parser.add_argument('--wandb_name', default=WANDB_NAME,
                    help='Name W&B will use when saving results.\ne.g. `--wandb_name "coco2017-full-sparse"`')

parser.add_argument('--wandb_entity', default=None,
                    help='(optional) Name of W&B team/entity to log to.')

parser.add_argument('--stable_softmax', dest='stable_softmax', action='store_true',
                    help='Prevent values from becoming too large during softmax. '
                         'Helps with stability in fp16 and Mixture of Quantization training.')

parser = distributed_utils.wrap_arg_parser(parser)

train_group = parser.add_argument_group('Training settings')

train_group.add_argument('--flops_profiler', dest='flops_profiler', action='store_true',
                         help='Exits after printing detailed flops/runtime analysis of forward/backward')

train_group.add_argument('--epochs', default=EPOCH_COUNT, type=int, help='Number of epochs')

train_group.add_argument('--save_every_n_steps', default=1000, type=int, help='Save a checkpoint every n steps')

train_group.add_argument('--keep_n_checkpoints', default=None, type=int,
                         help='(Careful) Deletes old deepspeed checkpoints if there are more than n')

train_group.add_argument('--batch_size', default=START_BATCH_SIZE, type=int, help='Batch size')

train_group.add_argument('--ga_steps', default=1, type=int,
                         help='Number of steps to accumulate gradients across per each iteration. DeepSpeed only.')

train_group.add_argument('--learning_rate', default=LEARNING_RATE, type=float, help='Learning rate')

train_group.add_argument('--clip_grad_norm', default=CLIP_GRAD_NORM, type=float, help='Clip gradient norm')

train_group.add_argument('--lr_decay', dest='lr_decay', action='store_true')

model_group = parser.add_argument_group('Model settings')

model_group.add_argument('--dim', default=512, type=int, help='Model dimension')

model_group.add_argument('--text_seq_len', default=256, type=int, help='Text sequence length')

model_group.add_argument('--depth', default=DEPTH_COUNT, type=int, help='Model depth') #default=2

model_group.add_argument('--heads', default=START_HEADS, type=int, help='Model number of heads')

model_group.add_argument('--dim_head', default=MODEL_DIM_HEAD, type=int, help='Model head dimension')

train_group.add_argument('--ff_dropout', default=F_DROPOUT, type=float, help='Feed forward dropout.')

train_group.add_argument('--attn_dropout', default=A_DROPOUT, type=float, help='Feed forward dropout.')

model_group.add_argument('--reversible', dest='reversible', action='store_true')

model_group.add_argument('--loss_img_weight', default=IMG_LOSS_WEIGHT, type=int, help='Image loss weight')

model_group.add_argument('--attn_types', default='full', type=str,
                         help='comma separated list of attention types. '
                              'attention type can be: full or sparse or axial_row or axial_col or conv_like.')

model_group.add_argument('--shift_tokens', help='Use the shift tokens feature', action='store_true')

model_group.add_argument('--rotary_emb', help='Use rotary embeddings', action='store_true')

model_group.add_argument('--shared_attn_ids', default=None, type=str,
                         help='Comma separated list of shared attention layer ids. Default: sharing is disabled')

model_group.add_argument('--shared_ff_ids', default=None, type=str,
                         help='Comma separated list of shared feed forward layer ids. Default: sharing is disabled')

model_group.add_argument('--share_input_output_emb', help='Share input and output embeddings', action='store_true')

args = parser.parse_args()


# helpers

def exists(val):
    return val is not None


def get_trainable_params(model):
    return [params for params in model.parameters() if params.requires_grad]


def get_pkg_version():
    from pkg_resources import get_distribution
    return get_distribution('dalle_pytorch').version


def cp_path_to_dir(cp_path, tag):
    """Convert a checkpoint path to a directory with `tag` inserted.
    If `cp_path` is already a directory, return it unchanged.
    """
    if not isinstance(cp_path, Path):
        cp_path = Path(cp_path)
    if cp_path.is_dir():
        return cp_path
    path_sans_extension = cp_path.parent / cp_path.stem
    cp_dir = Path(f'{path_sans_extension}-{tag}-cp')
    return cp_dir


# constants

WEBDATASET_IMAGE_TEXT_COLUMNS = tuple(args.wds.split(','))
ENABLE_WEBDATASET = True if len(WEBDATASET_IMAGE_TEXT_COLUMNS) == 2 else False

DALLE_OUTPUT_FILE_NAME = args.dalle_output_file_name + ".pt"

VAE_PATH = args.vae_path
VQGAN_MODEL_PATH = args.vqgan_model_path
VQGAN_CONFIG_PATH = args.vqgan_config_path
DALLE_PATH = args.dalle_path
RESUME = exists(DALLE_PATH)

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size

LEARNING_RATE = args.learning_rate
GRAD_CLIP_NORM = args.clip_grad_norm
LR_DECAY = cfg['lr_decay'] if USE_CFG else args.lr_decay
SAVE_EVERY_N_STEPS = args.save_every_n_steps
KEEP_N_CHECKPOINTS = args.keep_n_checkpoints

MODEL_DIM = args.dim
TEXT_SEQ_LEN = args.text_seq_len
DEPTH = args.depth
HEADS = args.heads
DIM_HEAD = args.dim_head
REVERSIBLE = cfg['reversible'] if USE_CFG else args.reversible
LOSS_IMG_WEIGHT = args.loss_img_weight
FF_DROPOUT = args.ff_dropout
ATTN_DROPOUT = args.attn_dropout
STABLE = cfg['stable_softmax'] if USE_CFG else args.stable_softmax
SHIFT_TOKENS = cfg['shift_tokens'] if USE_CFG else args.shift_tokens
ROTARY_EMB = cfg['rotary_emb'] if USE_CFG else args.rotary_emb

FP_16 = cfg['fp16'] if USE_CFG else args.fp16

ATTN_TYPES = tuple(args.attn_types.split(','))
SHARED_ATTN_IDS = tuple(args.shared_attn_ids.split(',')) if exists(args.shared_attn_ids) else None
SHARED_FF_IDS = tuple(args.shared_ff_ids.split(',')) if exists(args.shared_ff_ids) else None
SHARE_INPUT_OUTPUT_EMB = args.share_input_output_emb

DEEPSPEED_CP_AUX_FILENAME = 'auxiliary.pt'

if not ENABLE_WEBDATASET:
    # quit early if you used the wrong folder name
    assert Path(args.image_text_folder).exists(), f'The path {args.image_text_folder} was not found.'
else:
    # quit early if no tar files were found
    if Path(args.image_text_folder).is_dir():
        DATASET = [str(p) for p in Path(args.image_text_folder).glob("**/*") if ".tar" in str(p).lower()]  # .name
        assert len(DATASET) > 0, 'The directory ({}) does not contain any WebDataset/.tar files.'.format(
            args.image_text_folder)
        print('Found {} WebDataset .tar(.gz) file(s) under given path {}!'.format(len(DATASET), args.image_text_folder))
    elif ('http://' in args.image_text_folder.lower()) | ('https://' in args.image_text_folder.lower()):
        DATASET = f"pipe:curl -L -s {args.image_text_folder} || true"
        print('Found {} http(s) link under given path!'.format(len(DATASET), args.image_text_folder))
    elif 'gs://' in args.image_text_folder.lower():
        DATASET = f"pipe:gsutil cat {args.image_text_folder} || true"
        print('Found {} GCS link under given path!'.format(len(DATASET), args.image_text_folder))
    elif '.tar' in args.image_text_folder:
        DATASET = args.image_text_folder
        print('Found WebDataset .tar(.gz) file under given path {}!'.format(args.image_text_folder))
    else:
        raise Exception('No folder, no .tar(.gz) and no url pointing to tar files provided under {}.'.format(
            args.image_text_folder))

# initialize distributed backend
distr_backend = distributed_utils.set_backend_from_args(args)
distr_backend.initialize()

using_deepspeed = \
    distributed_utils.using_backend(distributed_utils.DeepSpeedBackend)

is_root = distr_backend.is_root_worker()

# tokenizer

if exists(args.bpe_path):
    klass = HugTokenizer if args.hug else YttmTokenizer
    tokenizer = klass(args.bpe_path)
elif args.chinese:
    tokenizer = ChineseTokenizer()

# reconstitute vae

if RESUME:
    dalle_path = Path(DALLE_PATH)
    if using_deepspeed:
        cp_dir = cp_path_to_dir(dalle_path, 'ds')
        assert cp_dir.is_dir(), \
            f'DeepSpeed checkpoint directory {cp_dir} not found'
        dalle_path = cp_dir / DEEPSPEED_CP_AUX_FILENAME
    else:
        assert dalle_path.exists(), 'DALL-E model file does not exist'
    loaded_obj = torch.load(str(dalle_path), map_location='cpu')

    dalle_params, vae_params, weights = loaded_obj['hparams'], loaded_obj['vae_params'], loaded_obj['weights']
    opt_state = loaded_obj.get('opt_state')
    scheduler_state = loaded_obj.get('scheduler_state')

    if vae_params is not None:
        vae = DiscreteVAE(**vae_params)
    elif args.taming:
        vae = VQGanVAE(VQGAN_MODEL_PATH, VQGAN_CONFIG_PATH)
    else:
        vae = OpenAIDiscreteVAE()

    IMAGE_SIZE = vae.image_size
    resume_epoch = loaded_obj.get('epoch', 0)
    print_blue(f"resuming from epoch {resume_epoch}, target epoch: {resume_epoch + EPOCHS}")
else:
    if exists(VAE_PATH):
        vae_path = Path(VAE_PATH)
        assert vae_path.exists(), 'VAE model file does not exist'
        assert not vae_path.is_dir(), \
            ('Cannot load VAE model from directory; please use a '
             'standard *.pt checkpoint. '
             'Currently, merging a DeepSpeed-partitioned VAE into a DALLE '
             'model is not supported.')

        loaded_obj = torch.load(str(vae_path))

        vae_params, weights = loaded_obj['hparams'], loaded_obj['weights']

        vae = DiscreteVAE(**vae_params)
        vae.load_state_dict(weights)
    else:
        if is_root:
            print('using pretrained VAE for encoding images to tokens')
        vae_params = None

        if args.taming:
            vae = VQGanVAE(VQGAN_MODEL_PATH, VQGAN_CONFIG_PATH)
        else:
            vae = OpenAIDiscreteVAE()

    IMAGE_SIZE = vae.image_size

    dalle_params = dict(
        num_text_tokens=tokenizer.vocab_size,
        text_seq_len=TEXT_SEQ_LEN,
        dim=MODEL_DIM,
        depth=DEPTH,
        heads=HEADS,
        dim_head=DIM_HEAD,
        reversible=REVERSIBLE,
        loss_img_weight=LOSS_IMG_WEIGHT,
        attn_types=ATTN_TYPES,
        ff_dropout=FF_DROPOUT,
        attn_dropout=ATTN_DROPOUT,
        stable=STABLE,
        shift_tokens=SHIFT_TOKENS,
        rotary_emb=ROTARY_EMB,
        shared_attn_ids=SHARED_ATTN_IDS,
        shared_ff_ids=SHARED_FF_IDS,
        share_input_output_emb=SHARE_INPUT_OUTPUT_EMB,
    )
    resume_epoch = 0

# configure OpenAI VAE for float16s

if isinstance(vae, OpenAIDiscreteVAE) and FP_16:
    vae.enc.blocks.output.conv.use_float16 = True


# helpers

def group_weight(model):
    group_decay, group_no_decay = [], []
    for params in model.named_parameters():
        if 'transformer' in params[0]:
            if 'bias' in params[0] or 'norm' in params[0]:
                group_no_decay.append(params[1])
                continue
        group_decay.append(params[1])

    assert len(list(model.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


# create dataset and dataloader

is_shuffle = not distributed_utils.using_backend(distributed_utils.HorovodBackend)

imagepreproc = T.Compose([
    T.Lambda(lambda img: img.convert('RGB')
    if img.mode != 'RGB' else img),
    T.RandomResizedCrop(IMAGE_SIZE,
                        scale=(args.resize_ratio, 1.),
                        ratio=(1., 1.)),
    T.ToTensor(),
])


def imagetransform(b):
    return Image.open(BytesIO(b))


def tokenize(s):
    return tokenizer.tokenize(
        s.decode('utf-8'),
        TEXT_SEQ_LEN,
        truncate_text= cfg['truncate_captions'] if USE_CFG else args.truncate_captions
    ).squeeze(0)


if ENABLE_WEBDATASET:
    DATASET_SIZE = int(
        1e9)  # You need to set a nominal length for the Dataset in order to avoid warnings from DataLoader

    myimg, mycap = WEBDATASET_IMAGE_TEXT_COLUMNS
    image_text_mapping = {
        myimg: imagetransform,
        mycap: tokenize
    }
    image_mapping = {
        myimg: imagepreproc
    }


    def filter_dataset(item):  # For e.g. C@H which (rarely) has no caption available.
        if mycap not in item:
            return False
        if myimg not in item:
            return False
        return True


    w_dataset = wds.WebDataset(DATASET, handler=wds.warn_and_continue)
    filtered_dataset = w_dataset.select(filter_dataset)
    ds = filtered_dataset.map_dict(**image_text_mapping).map_dict(**image_mapping).to_tuple(mycap, myimg).batched(
        BATCH_SIZE / distr_backend.get_world_size(), partial=True)
else:
    ds = TextImageDataset(
        args.image_text_folder,
        text_len=TEXT_SEQ_LEN,
        image_size=IMAGE_SIZE,
        resize_ratio=args.resize_ratio,
        truncate_captions=args.truncate_captions,
        tokenizer=tokenizer,
        shuffle=is_shuffle,
    )
    assert len(ds) > 0, 'dataset is empty'

if is_root:
    if not ENABLE_WEBDATASET:
        print(f'{len(ds)} image-text pairs found for training')

# data sampler

data_sampler = None

if not is_shuffle:
    data_sampler = torch.utils.data.distributed.DistributedSampler(
        ds,
        num_replicas=distr_backend.get_world_size(),
        rank=distr_backend.get_rank()
    )

# WebLoader for WebDataset and DeepSpeed compatibility

if ENABLE_WEBDATASET:
    dl = wds.WebLoader(ds, batch_size=None, shuffle=False, num_workers=4)  # optionally add num_workers=2 (n) argument
    number_of_batches = DATASET_SIZE // (BATCH_SIZE * distr_backend.get_world_size())
    dl = dl.slice(number_of_batches)
    dl.length = number_of_batches
else:
    # Regular DataLoader for image-text-folder datasets
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=is_shuffle, drop_last=True, sampler=data_sampler)

# initialize DALL-E

dalle = DALLE(vae=vae, **dalle_params)

if not using_deepspeed:
    if FP_16:
        dalle = dalle.half()
    dalle = dalle.cuda()
    if RESUME:
        dalle.load_state_dict(weights)

# optimizer

if OPTIMIZER == ADAMAX:
    opt = Adamax(get_trainable_params(dalle),
                 lr=LEARNING_RATE,
                 betas=(cfg['beta1'], cfg['beta2']) if USE_CFG else (0.9, 0.999),
                 weight_decay=cfg['weight_decay'] if USE_CFG else 0,
                 eps=cfg['epsilon'] if USE_CFG else 1e-8
                 )
elif OPTIMIZER == ADAMW:
    opt = AdamW(get_trainable_params(dalle),
                lr=LEARNING_RATE,
                betas=(cfg['beta1'], cfg['beta2']) if USE_CFG else (0.9, 0.999),
                weight_decay=cfg['weight_decay'] if USE_CFG else 0,
                eps=cfg['epsilon'] if USE_CFG else 1e-8,
                amsgrad=cfg['amsgrad'] if USE_CFG else False
                )
elif OPTIMIZER == RADAM:
    opt = RAdam(get_trainable_params(dalle),
                lr=LEARNING_RATE,
                betas=(cfg['beta1'], cfg['beta2']) if USE_CFG else (0.9, 0.999),
                weight_decay=cfg['weight_decay'] if USE_CFG else 0,
                eps=cfg['epsilon'] if USE_CFG else 1e-8
                )
elif OPTIMIZER == NADAM:
    opt = NAdam(get_trainable_params(dalle),
                lr=LEARNING_RATE,
                betas=(cfg['beta1'], cfg['beta2']) if USE_CFG else (0.9, 0.999),
                weight_decay=cfg['weight_decay'] if USE_CFG else 0,
                eps=cfg['epsilon'] if USE_CFG else 1e-8
                )
else:
    opt = Adam(
        get_trainable_params(dalle),
        lr=LEARNING_RATE,
        betas=(cfg['beta1'], cfg['beta2']) if USE_CFG else (0.9, 0.999),
        weight_decay=cfg['weight_decay'] if USE_CFG else 0,
        eps=float(cfg['epsilon']) if USE_CFG else 1e-8,
        amsgrad=cfg['amsgrad'] if USE_CFG else False
    )

print_blue(f"using optimizer {OPTIMIZER}\nlearning rate: {LEARNING_RATE}\n")

if RESUME and opt_state:
    opt.load_state_dict(opt_state)

# scheduler

scheduler = None

if LR_DECAY:
    scheduler = ReduceLROnPlateau(
        opt,
        mode="min",
        factor=cfg['decay_factor'] if USE_CFG else 0.5,
        patience=cfg['patience'] if USE_CFG else 10,
        cooldown=cfg['cooldown'] if USE_CFG else 10,
        min_lr=float(cfg['min_lr']) if USE_CFG else 1e-6,
        threshold=float(cfg['threshold']) if USE_CFG else 1e-4,
        eps=float(cfg['eps']) if USE_CFG else 1e-8,
        verbose=True,
    )
    if RESUME and scheduler_state:
        scheduler.load_state_dict(scheduler_state)

# experiment tracker

if is_root:
    model_config = dict(
        depth=DEPTH,
        heads=HEADS,
        dim_head=DIM_HEAD
    )

    run = wandb.init(
        project=args.wandb_name,
        entity=args.wandb_entity,
        resume=False,
        config=model_config,
    )

# distribute

distr_backend.check_batch_size(BATCH_SIZE)
deepspeed_config = {
    'train_batch_size': BATCH_SIZE,
    'gradient_accumulation_steps': args.ga_steps,
    'gradient_clipping': GRAD_CLIP_NORM,
    'fp16': {
        'enabled': FP_16,
    },
    'amp': {
        'enabled': cfg['amp'] if USE_CFG else args.amp,
        'opt_level': 'O1',
    },
    "flops_profiler": {
        "enabled": args.flops_profiler,
        "profile_step": 200,
        "module_depth": -1,
        "top_modules": 1,
        "detailed": True,
        "output_file": None  # TODO Can't get this to work.
    },
}

if deepspeed_config.get('zero_optimization', {}).get('stage', 0) >= 2:
    print(f"Checkpoints made with DeepSpeed ZeRO Stages 2 and 3 will be stored in deepspeed checkpoint folder")
    print(f"As such, they will require DeepSpeed as a dependency in order to resume from or generate with.")
    print(
        "See the deespeed conversion script for details on how to convert your ZeRO stage 2/3 checkpoint to a single file.")
    print(
        "If using a single GPU, consider running with apex automatic mixed precision instead for a similar speedup to ZeRO.")
    time.sleep(2)

(distr_dalle, distr_opt, distr_dl, distr_scheduler) = distr_backend.distribute(
    args=args,
    model=dalle,
    optimizer=opt,
    model_parameters=get_trainable_params(dalle),
    training_data=(
        (None if ENABLE_WEBDATASET else ds)
        if using_deepspeed
        else dl
    ),
    # Do not pass the LR scheduler to DeepSpeed so we can manually
    # advance it.
    lr_scheduler=scheduler if LR_DECAY and not using_deepspeed else None,
    config_params=deepspeed_config,
)
# Prefer scheduler in `deepspeed_config`.

if LR_DECAY and distr_scheduler is None:
    distr_scheduler = scheduler

avoid_model_calls = using_deepspeed and FP_16

if RESUME and using_deepspeed:
    distr_dalle.load_checkpoint(str(cp_dir))


def save_model(path, epoch=0):
    save_obj = {
        'hparams': dalle_params,
        'vae_params': vae_params,
        'epoch': epoch,
        'version': get_pkg_version(),
        'vae_class_name': vae.__class__.__name__
    }

    if using_deepspeed:
        cp_dir = cp_path_to_dir(path, 'ds')

        if KEEP_N_CHECKPOINTS is not None and is_root:
            checkpoints = sorted(glob(str(cp_dir / "global*")), key=os.path.getmtime, reverse=True)
            for checkpoint in checkpoints[KEEP_N_CHECKPOINTS:]:
                shutil.rmtree(checkpoint)

        distr_dalle.save_checkpoint(cp_dir, client_state=save_obj)

        if not is_root:
            return

        # Save auxiliary values so we can reuse the standard routine
        # for loading.
        save_obj = {
            **save_obj,
            # Save a nonsense value that directs the user to
            # further help.
            'weights': (
                'To get a working standard checkpoint, '
                'look into consolidating DeepSpeed checkpoints.'
            ),
        }
        torch.save(save_obj, str(cp_dir / DEEPSPEED_CP_AUX_FILENAME))
        if deepspeed_config.get('zero_optimization', {}).get('stage',
                                                             0) >= 2:  # see https://github.com/lucidrains/DALLE-pytorch/wiki/DeepSpeed-Checkpoints
            return

    if not is_root:
        return

    save_obj = {
        **save_obj,
        'weights': dalle.state_dict(),
        'opt_state': opt.state_dict(),
        'scheduler_state': (scheduler.state_dict() if scheduler else None)
    }

    torch.save(save_obj, path)


def save_artifact(model_config, model_path, name='trained-dalle'):
    model_artifact = wandb.Artifact(name, type='model', metadata=dict(model_config))
    model_artifact.add_file(model_path)
    run.log_artifact(model_artifact)


# training

# Saves a checkpoint before training begins to fail early when mis-configured.
# See https://github.com/lucidrains/DALLE-pytorch/wiki/DeepSpeed-Checkpoints

save_model(DALLE_OUTPUT_FILE_NAME, epoch=resume_epoch)
current_epoch = resume_epoch
target_epoch = current_epoch + EPOCHS

topk_index = 0
temp_index = 0

for epoch in tqdm(range(resume_epoch, target_epoch), colour='green'):
    if data_sampler:
        data_sampler.set_epoch(epoch)

    for i, (text, images) in enumerate((dl if ENABLE_WEBDATASET else distr_dl)):
        if i % 10 == 0 and is_root:
            t = time.time()

        if FP_16:
            images = images.half()

        text, images = map(lambda t: t.cuda(), (text, images))

        loss = distr_dalle(text, images, return_loss=True)

        if using_deepspeed:
            distr_dalle.backward(loss)
            distr_dalle.step()
            # Gradients are automatically zeroed after the step
        else:
            loss.backward()
            clip_grad_norm_(distr_dalle.parameters(), GRAD_CLIP_NORM)
            distr_opt.step()
            distr_opt.zero_grad()

        # Collective loss, averaged
        avg_loss = distr_backend.average_all(loss)

        log = {}

        if i % 10 == 0 and is_root:
            GAP = ' ' if i < 100 else ''
            if i == 0:
                GAP = '  '
                print_blue(f'\n{epoch} {GAP}{i} loss - {avg_loss.item()}')
            else:
                print_blue(f'{epoch} {GAP}{i} loss - {avg_loss.item()}')
            log = {
                **log,
                'epoch': epoch,
                'iter': i,
                'loss': avg_loss.item()
            }

        if i % SAVE_EVERY_N_STEPS == 0:
            save_model(DALLE_OUTPUT_FILE_NAME, epoch=epoch)

        if i % GEN_IMG_STEPS == 0 and is_root:
            sample_text = text[:1]
            token_list = sample_text.masked_select(sample_text != 0).tolist()
            decoded_text = tokenizer.decode(token_list)

            if not avoid_model_calls:
                image = dalle.generate_images(
                    text[:1],
                    filter_thres=TOP_K[min(topk_index, len(TOP_K) - 1)],
                    temperature=TEMPERATURE[min(temp_index, len(TEMPERATURE) - 1)]
                )
                log['image'] = wandb.Image(image, caption=decoded_text)
                topk_index = topk_index + 1
                if topk_index > len(TOP_K) - 1:
                    topk_index = 0
                temp_index = temp_index + 1
                if temp_index > len(TEMPERATURE) - 1:
                    temp_index = 0

        if i % 10 == 9 and is_root:
            sample_per_sec = BATCH_SIZE * 10 / (time.time() - t)
            log["sample_per_sec"] = sample_per_sec
            if not USE_CFG or (USE_CFG and cfg['verbose_sample_per_sec']):
                print(f'{epoch} {i} sample_per_sec - {sample_per_sec}')

        if i == 201 and args.flops_profiler:
            raise StopIteration("Profiler has finished running. Stopping training early.")

        if is_root:
            wandb.log(log)

        if LR_DECAY:
            distr_scheduler.step(avg_loss)

    save_model(DALLE_OUTPUT_FILE_NAME, epoch=epoch)

    if is_root:
        # save trained model to wandb as an artifact every epoch's end
        save_artifact(model_config, DALLE_OUTPUT_FILE_NAME)

    current_epoch = epoch


save_model(DALLE_OUTPUT_FILE_NAME, epoch=current_epoch)
print_blue(f"Saved {DALLE_OUTPUT_FILE_NAME} after epoch {current_epoch}")

if is_root:
    wandb.save(DALLE_OUTPUT_FILE_NAME)
    save_artifact(model_config, DALLE_OUTPUT_FILE_NAME)
    wandb.finish()
