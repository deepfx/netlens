import torch
import numpy as np

class Callback():
    def __init__(self, learn):
        self.learn = learn

    def on_train_begin(self, **kwargs):
        pass

    def on_epoch_begin(self, **kwargs):
        pass

    def on_batch_begin(self, **kwargs):
        pass

    def on_batch_end(self, **kwargs):
        pass

    def on_epoch_end(self, **kwargs):
        pass

    def on_train_end(self, **kwargs):
        pass

class OneCycleSchedule(Callback):
    def __init__(self, learn, max_lr, moms=(0.95, 0.85), div_factor=25, pct_start=0.3):
        self.learn = learn
        self.max_lr = max_lr
        self.moms = moms
        self.div_factor = div_factor
        self.pct_start = pct_start
        self.batch_len = len(learn.data)
        self.step = 0

        cos_vals = np.arange(0, np.pi, np.pi/round(self.batch_len*(1-pct_start)))
        if isinstance(max_lr, (tuple, list)):
            self.lr_sched = [np.concatenate([np.linspace(lr/25, lr, round(self.batch_len*pct_start)),
                                        (np.cos(cos_vals)+1)*lr/2]) for lr in max_lr]
        else:
            self.lr_sched = np.concatenate([np.linspace(max_lr/25, max_lr, round(self.batch_len*pct_start)),
                                        (np.cos(cos_vals)+1)*max_lr/2])
        self.mom_sched = np.concatenate([np.linspace(moms[0], moms[1], round(self.batch_len*pct_start)),
                                        (np.cos(cos_vals[::-1])+1)*(moms[0]-moms[1])/2+moms[1]])

    def on_batch_begin(self):
        for i,pg in enumerate(self.learn.optim.param_groups):
            if isinstance(self.max_lr, float):
                pg['lr'] = self.lr_sched[self.step]
            else:
                pg['lr'] = self.lr_sched[i][self.step]
            beta1,beta2 = pg['betas']
            pg['betas'] = self.mom_sched[self.step],beta2
        self.step += 1

    def on_epoch_end(self):
        self.step = 0

class Recorder(Callback):
    def __init__(self, learn):
        self.learn = learn
        self.lrs = []
        self.moms = []

    def on_batch_begin(self):
        self.lrs.append(self.learn.optim.param_groups[0]['lr'])
        self.moms.append(self.learn.optim.param_groups[0]['betas'][0])
