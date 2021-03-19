import copy


class AttrDict(dict):
    def __init__(self, kwargs={}):
        super().__init__(kwargs)
        self.__dict__ = self
        
    def __copy__(self):
        cls = self.__class__
        result = cls()
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls()
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

config = AttrDict()

config.diffusion = AttrDict()
config.diffusion.name = 'diffusion'
config.diffusion.save_dir = '../generated_files'
config.diffusion.dataset = AttrDict({
    'name': 'ffhq',
    'path': '../data/faces',
    'resolution': 128
})
config.diffusion.model = AttrDict({
    'in_channel': 3,
    'channel': 128,
    'channel_multiplier': [1, 1, 2, 2, 4, 4],
    'n_res_blocks': 2,
    'attn_strides': [16],
    'attn_heads': 1,
    'use_affine_time': False,
    'dropout': 0.0,
    'fold': 1
})
config.diffusion.diffusion = AttrDict({
    'beta_schedule': AttrDict({
        'schedule': 'linear',
        'n_timestep': 1000,
        'linear_start': 1e-4,
        'linear_end': 2e-2
    })
})
config.diffusion.training = AttrDict({
    'n_iter': 1000000,
    'optimizer': AttrDict({
        'type': 'adam',
        'lr': 2e-5
    }),
    'scheduler': AttrDict({
        'type': 'cycle',
        'lr': 2e-5,
        'n_iter': 1000000,
        'warmup': 5000,
        'decay': ['linear', 'flat']
    }),
    'dataloader': AttrDict({
        'batch_size': 16,
        'num_workers': 8,
        'drop_last': True
    })
})
config.diffusion.evaluate = AttrDict({
    'save_every': 5000,
    'sample_every': 1000
})

config['diffusion-m'] = copy.deepcopy(config.diffusion)
config['diffusion-m'].name += '-m'
config['diffusion-m'].model.update({
    'channel': 64,
    'channel_multiplier': [1, 2, 2, 4, 4, 8],
    'n_res_blocks': 1
})

config['diffusion-s'] = copy.deepcopy(config.diffusion)
config['diffusion-s'].name += '-s'
config['diffusion-s'].model.update({
    'channel': 32,
    'channel_multiplier': [1, 2, 4, 4, 8],
    'n_res_blocks': 1,
    'attn_strides': [],
})

config.improved = copy.deepcopy(config.diffusion)
config.improved.name = 'diffusion_improved'
config.improved.model.update({
    'attn_strides': [8, 16],
    'attn_heads': 4,
    'use_affine_time': True
})
config.improved.diffusion.beta_schedule.update({
    'schedule': 'cosine',
    'cosine_s': 8e-3
})
config.improved.training.n_iter = 1000000
config.improved.training.optimizer.update({
    'lr': 5e-5
})
config.improved.training.scheduler.update({
    'lr': 5e-5
})

config['improved-s'] = copy.deepcopy(config.improved)
config['improved-s'].name += '-s'
config['improved-s'].model.update({
    'channel': 32,
    'channel_multiplier': [1, 2, 4, 4, 8],
    'n_res_blocks': 1,
    'attn_strides': []
})
config['improved-s'].training.optimizer.update({
    'lr': 2e-4
})
config['improved-s'].training.scheduler.update({
    'lr': 2e-4
})
config['improved-s'].training.dataloader.update({
    'batch_size': 128
})
