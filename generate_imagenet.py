import argparse
import os
import shutil
from functools import partial

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from torch.utils import data
from torch.utils.data import DistributedSampler, TensorDataset
import torch.multiprocessing as mp
from torchvision.utils import save_image
from tqdm import tqdm

import core.util as Util
from diffusion import create_diffusion
from download import find_model
from models import DiT_XL_2
from diffusers.models import AutoencoderKL


def main_worker(gpu, ngpus_per_node, opt):
    if 'local_rank' not in opt:
        opt.local_rank = opt.global_rank = gpu
    if opt.distributed:
        torch.cuda.set_device(int(opt.local_rank))
        print('using GPU {} for generation'.format(int(opt.local_rank)))
        torch.distributed.init_process_group(backend = 'nccl',
                                             init_method = opt.init_method,
                                             world_size = opt.world_size,
                                             rank = opt.global_rank,
                                             group_name='mtorch'
                                             )
    set_device = partial(Util.set_device, rank=opt.global_rank)
    '''set seed and and cuDNN environment '''
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)
    num_sampling_steps = opt.steps
    # Nested Diffusion skips the last sampling step from $t = 1$ to $t = 0$, as this is computed in the last inner step
    num_sampling_steps = num_sampling_steps + 1 if opt.nested is not None else num_sampling_steps
    cfg_scale = opt.cfg

    # Load model:
    image_size = 256
    latent_channels = 4
    assert image_size in [256, 512], "We only provide pre-trained models for 256x256 and 512x512 resolutions."
    latent_size = image_size // 8
    model = DiT_XL_2(input_size=latent_size)
    state_dict = find_model(f"DiT-XL-2-{image_size}x{image_size}.pt")
    model.load_state_dict(state_dict)
    model = set_device(model, distributed=opt.distributed)
    model.eval()
    diffusion = create_diffusion(str(num_sampling_steps))
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema")
    vae = set_device(vae, distributed=opt.distributed)
    vae.eval()

    images_in_output = opt.nested if opt.nested is not None else 1


    if opt.global_rank == 0:
        if os.path.exists(opt.output):
            shutil.rmtree(opt.output)
        os.makedirs(opt.output, exist_ok=True)
        for i in range(images_in_output):
            os.makedirs(os.path.join(opt.output, f"{(opt.nested if opt.nested is not None else 1) * i}"), exist_ok=True)
        if opt.distributed:
            torch.distributed.barrier()
    else:
        if opt.distributed:
            torch.distributed.barrier()

    assert opt.number % 1000 == 0
    per_class = (opt.number // 1000)

    dataset = TensorDataset(torch.arange(1000).repeat_interleave(per_class), torch.arange(1000 * per_class))
    data_sampler = None
    if opt.distributed:
        data_sampler = DistributedSampler(dataset, shuffle=False, num_replicas=opt.world_size, rank=opt.global_rank)
    class_loader = data.DataLoader(
        dataset,
        sampler=data_sampler,
        batch_size=opt.batch,
        shuffle=False,
        num_workers=opt.num_workers,
        drop_last=False
    )

    for class_, n in tqdm(class_loader):
        b = class_.shape[0]
        z = torch.randn(b, latent_channels, latent_size, latent_size)
        y = class_.clone().int()
        z, y = set_device(z, distributed=opt.distributed), set_device(y, distributed=opt.distributed)
        func = (model.module.forward_with_cfg if opt.distributed else model.forward_with_cfg) if cfg_scale != 1.0 else model
        model_kwargs = dict(y=y, cfg_scale=cfg_scale) if cfg_scale != 1.0 else dict(y=y)

        if opt.nested:
            samples = diffusion.nested_sample_loop(func, z.shape, z, inner_steps=opt.nested,
                                                   ddim=opt.ddim, clip_denoised=False,
                                                   model_kwargs=model_kwargs, progress=True, device=z.device)
        elif opt.ddim:
            samples = diffusion.ddim_sample_loop(func, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs,
                                                 progress=True, eta=opt.eta, device=z.device)
        else:
            samples = diffusion.p_sample_loop(func, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs,
                                              progress=True, device=z.device)

        decode = vae.module.decode if opt.distributed else vae.decode
        if not isinstance(samples, list):
            samples = [samples]
        samples = [decode(s / 0.18215).sample for s in samples[:-1]]
        for i, sample in enumerate(samples):
            for j in range(b):
                number = n[j].item()
                save_image((0.5 + 0.5 * sample[j]),
                           os.path.join(opt.output,
                                        f"{(opt.nested if opt.nested is not None else 1) * i}",
                                        f"{number:06d}.png"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--number', type=int, default=10000)
    parser.add_argument('-b', '--batch', type=int, default=64)
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-s', '--steps', type=int, default=50)
    parser.add_argument('-e', '--eta', type=float, default=1.0)
    parser.add_argument('--nested', type=int, default=0)
    parser.add_argument('--ddim', action='store_true', default=False)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--cfg', type=float, default=1.0)
    parser.add_argument('-P', '--port', default='21012', type=str)
    parser.add_argument('-gpu', '--gpu_ids', type=str, default="0")

    ''' parser configs '''
    args = parser.parse_args()

    ''' cuda devices '''
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    print('export CUDA_VISIBLE_DEVICES={}'.format(args.gpu_ids))
    args.gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    args.distributed = len(args.gpu_ids) > 1

    ''' use DistributedDataParallel(DDP) and multiprocessing for multi-gpu training'''
    if args.distributed:
        ngpus_per_node = len(args.gpu_ids)  # or torch.cuda.device_count()
        args.world_size = ngpus_per_node
        args.init_method = 'tcp://127.0.0.1:' + args.port
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        args.world_size = 1
        main_worker(0, 1, args)
