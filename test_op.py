import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from model import UNet
from diffusion import make_beta_schedule, GaussianDiffusion
from config import config

conf = config.diffusion
betas = make_beta_schedule(**conf.diffusion.beta_schedule)
diffusion = GaussianDiffusion(betas)
model = UNet(**conf.model)
img = paddle.randn([
    conf.training.dataloader.batch_size,
    conf.model.in_channel,
    conf.dataset.resolution,
    conf.dataset.resolution
])
time = paddle.randint(
    0, conf.diffusion.beta_schedule.n_timestep, (img.shape[0],)
)
loss = diffusion.p_loss(model, img, time)
print(loss.numpy())


conf = config.improved
betas = make_beta_schedule(**conf.diffusion.beta_schedule)
diffusion = GaussianDiffusion(betas)
model = UNet(**conf.model)
img = paddle.randn([
    conf.training.dataloader.batch_size,
    conf.model.in_channel,
    conf.dataset.resolution,
    conf.dataset.resolution
])
time = paddle.randint(
    0, conf.diffusion.beta_schedule.n_timestep, (img.shape[0],)
)
loss = diffusion.p_loss(model, img, time)
loss.backward()
print(loss.numpy())