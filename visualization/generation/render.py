import torch.optim


class OptVis:

    def __init__(self, model, objective, tfms=None, optim=torch.optim.Adam, optim_params=None):
        self.model = model
        self.objective = objective
        self.tfms = tfms
        self.optim_fn = optim
        self.optim_params = optim_params or {}
        print(f"Optimizing for {objective}")
        self.model.eval()

        # state variables
        self.optim = None
        self.run = 0

    def vis(self, img_param, thresh=(100,), transform=True, in_closure=False, verbose=True, callback=None, display=False):
        self.optim = self.optim_fn(img_param.parameters(), **self.optim_params)
        self.run = 0

        if callback:
            callback.on_render_begin(self, img_param)

        def closure():
            self.optim.zero_grad()

            _loss = self.objective(img)
            _loss.backward()

            self.run += 1
            if verbose and self.run % 50 == 0:
                print(f'Run [{self.run}], loss={_loss.item():.4f}')

            return _loss

        while self.run <= max(thresh):
            img = img_param()

            if transform:
                img = self.tfms(img)

            if callback:
                callback.on_step_begin(self, img)

            if in_closure:
                self.optim.step(closure)
            else:
                # "normal" case, without closure
                closure()
                self.optim.step()

            if callback:
                callback.on_step_end(self, img)

        # get the last image, for final tasks and returning it
        img = img_param()

        if callback:
            callback.on_render_end(self, img)

        return img


class OptVisCallback:

    def on_render_begin(self, optvis, *args, **kwargs):
        pass

    def on_step_begin(self, optvis, img, *args, **kwargs):
        pass

    def on_step_end(self, optvis, img, *args, **kwargs):
        pass

    def on_render_end(self, optvis, img, *args, **kwargs):
        pass
