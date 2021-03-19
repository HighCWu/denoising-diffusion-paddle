import os
import copy
import argparse

import paddle
from paddle import nn
from paddle.nn import functional as F
from paddle.vision import transforms, datasets
from paddle.io import DataLoader

import numpy as np
from PIL import Image

from tqdm import tqdm

from model import UNet
from diffusion import make_beta_schedule, GaussianDiffusion
from config import config

import lr_scheduler

dirname = os.path.basename(os.path.dirname(os.path.abspath(__file__)))


def sample_data(loader):
    loader_iter = iter(loader)
    epoch = 0

    while True:
        try:
            yield epoch, next(loader_iter)

        except StopIteration:
            epoch += 1
            loader_iter = iter(loader)

            yield epoch, next(loader_iter)


@paddle.no_grad()
def accumulate(model1, model2, decay=0.9999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k][:] = par1[k] * decay + par2[k] * (1 - decay)


def train(conf, loader, model, ema, diffusion, optimizer, scheduler):
    loader = sample_data(loader)

    pbar = tqdm(range(conf.training.n_iter + 1))

    noise_img = None
    sample_size = 4
    grid_size = 5

    for i in pbar:
        epoch, img_cls = next(loader)
        img = img_cls[0]
        if noise_img is None:
            noise_img = paddle.randn([sample_size, *img.shape[1:]])
        time = paddle.randint(
            0, conf.diffusion.beta_schedule.n_timestep, (img.shape[0],)
        )
        loss = diffusion.p_loss(model, img, time)
        optimizer.clear_grad()
        loss.backward()
        scheduler.step()
        optimizer.step()

        accumulate(
            ema, model, 0 if i < conf.training.scheduler.warmup else 0.9999
        )

        lr = optimizer.get_lr()
        pbar.set_description(
            f"epoch: {epoch}; loss: {loss.numpy()[0]:.4f}; lr: {lr:.6f}"
        )

        if i % conf.evaluate.sample_every == 0:
            ema.eval()
            
            img = noise_img
            n_timestep = conf.diffusion.beta_schedule.n_timestep
            sample_idx = [0] + [i * n_timestep // grid_size**2 for i in range(1,grid_size**2-1)] + [n_timestep-1]
            imgs = []

            with paddle.no_grad():
                print('\n')
                pbar2 = tqdm(list(reversed(range(n_timestep))))
                pbar2.set_description("Image sampling...")
                for j in pbar2:
                    img = diffusion.p_sample(
                        ema,
                        img,
                        paddle.full((sample_size,), j, dtype=np.int64),
                        noise_fn=paddle.randn,
                    )

                    if j == n_timestep-1:
                        imgs.append(noise_img)
                    elif j in sample_idx:
                        imgs.append(img)

                    if j == 0:
                        pbar2.set_description("Image sampling finished.")
                        print('\n')

                imgs = paddle.stack(imgs, 1)

                pbar3 = tqdm(list(imgs))
                pbar3.set_description("Image saving...")
                for k, img in enumerate(pbar3):
                    out_img = paddle.reshape(img, [grid_size, grid_size, *img.shape[1:]])
                    out_img = paddle.transpose(out_img, [0,3,1,4,2])
                    out_img = paddle.reshape(out_img, [out_img.shape[1]*grid_size,-1,out_img.shape[-1]])
                    out_img = (out_img + 1) / 2 * 255
                    out_img = paddle.clip(paddle.round(out_img), 0, 255)
                    out_img = paddle.cast(out_img, 'uint8')
                    out_img = out_img.numpy()
                    out_img = Image.fromarray(out_img)
                    out_img.save(f'{conf.full_save_dir}/sample/{str(i).zfill(5)}_{str(k).zfill(3)}.png')

                    if k == len(pbar3) - 1:
                        pbar3.set_description("Image saving finished.")
                        print('\n')

        if i % conf.evaluate.save_every == 0:
            print('\nSaving model...')
            paddle.save(model.state_dict(), f'{conf.full_save_dir}/checkpoint/model.pdparams')
            paddle.save(ema.state_dict(), f'{conf.full_save_dir}/checkpoint/ema.pdparams')
            paddle.save(optimizer.state_dict(), f'{conf.full_save_dir}/checkpoint/optimizer.opt')
            paddle.save(scheduler.state_dict(), f'{conf.full_save_dir}/checkpoint/scheduler.sche')
            paddle.save(conf, f'{conf.full_save_dir}/checkpoint/conf.pkl')
            print('Model Saved.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', 
        type=str, 
        default='improved-s', 
        help='diffusion/diffusion-m/diffusion-s/improved/improved-s'
    )

    args, _ = parser.parse_known_args()

    print(args)

    conf = config[args.config]

    full_save_dir = f'{conf.save_dir}/{dirname}/{conf.name}'
    conf.full_save_dir = full_save_dir

    os.makedirs(f'{full_save_dir}/sample', exist_ok=True)
    os.makedirs(f'{full_save_dir}/checkpoint', exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Resize([conf.dataset.resolution]*2),
            transforms.Transpose(),
            transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5)),
        ]
    )

    train_dataset = datasets.ImageFolder(conf.dataset.path, transform=transform)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=conf.training.dataloader.batch_size, 
        shuffle=True, 
        num_workers=conf.training.dataloader.num_workers, 
        drop_last=conf.training.dataloader.drop_last,
        use_shared_memory=False)
    
    model = UNet(**conf.model)
    ema = UNet(**conf.model)
    ema.eval()

    clip = nn.ClipGradByNorm(clip_norm=1.0)
    optimizer_types = {
        'adam': paddle.optimizer.Adam, 
    }
    optimizer = optimizer_types[conf.training.optimizer.type](
        parameters=model.parameters(), 
        learning_rate=conf.training.optimizer.lr,
        grad_clip=clip
    )

    scheduler_conf = copy.deepcopy(conf.training.scheduler)
    scheduler_types = {
        'cycle': lr_scheduler.cycle_scheduler,
        'step': lr_scheduler.step_scheduler,
        'default': lr_scheduler.ConstantScheduler
    }
    scheduler_type = scheduler_conf.pop('type', 'default')
    scheduler = scheduler_types[scheduler_type](optimizer, **scheduler_conf)

    betas = make_beta_schedule(**conf.diffusion.beta_schedule)
    diffusion = GaussianDiffusion(betas)

    train(
        conf, train_loader, model, ema, diffusion, optimizer, scheduler
    )
