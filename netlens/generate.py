from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.optim import Adam, SGD

from netlens.image_proc import preprocess_image, recreate_image
from netlens.math import normalized_euclidean_loss, alpha_norm, total_variation_norm
from netlens.modules import FlatModel, MODELS_CONFIG


class NetDreamer:

    def __init__(self, raw_model: nn.Module, model: Optional[FlatModel] = None):
        self.raw_model = raw_model
        self.original_model = model if model is not None else FlatModel.from_nested_cnn(raw_model)
        self.model_input_size = MODELS_CONFIG['input_size'].get(self.original_model.arch_name, (224, 224))

        self.raw_model.eval()
        self.original_model.eval()

    def _prepare_model(self):
        self.model = self.original_model.copy()
        # Put model in evaluation mode
        self.model.eval()

    def _generate_random_image(self, low: int = 0, high: int = 256) -> torch.Tensor:
        random_image = np.random.uniform(low, high, self.model_input_size + (3,)).astype(np.uint8)
        return preprocess_image(random_image, resize_to=None)

    def generate_filter_visualization(self, target_layer: str, target_channel: int, num_iters: int = 30):
        self._prepare_model()

        # Generate a random image
        processed_image = self._generate_random_image(low=150, high=180)

        # Define optimizer for the image
        optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-6)
        for i in range(num_iters):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            out = self.model.forward(processed_image, until_layer=target_layer)
            # Here, we get the specific filter from the output of the convolution operation
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(out[0, target_channel])
            print(f'Iteration: {i + 1}, Loss: {loss.item():.4f}')
            # Backward
            loss.backward()
            # Update image
            optimizer.step()

        return recreate_image(processed_image), processed_image

    def generate_class_sample(self, target_class: int, num_iters: int = 150, lr: float = 6.0):
        processed_image = self._generate_random_image()

        optimizer = SGD([processed_image], lr=lr)

        for i in range(num_iters):
            # Define optimizer for the image
            # Forward --> use the raw model for faster processing
            output = self.raw_model(processed_image)
            # Target specific class
            class_loss = -output[0, target_class]
            print(f'Iteration: {i + 1}, Loss: {class_loss.item():.4f}')
            # Zero grads
            self.raw_model.zero_grad()
            # Backward
            class_loss.backward()
            # Update image
            optimizer.step()
            # Recreate image

        return recreate_image(processed_image), processed_image

    def deep_dream(self, input_image, target_layer: str, target_channel: int, num_iters: int = 250):
        self._prepare_model()

        processed_image = preprocess_image(input_image, resize_to=(512, 512))

        # Define optimizer for the image
        # Earlier layers need higher learning rates to visualize whereas layer layers need less
        optimizer = SGD([processed_image], lr=12, weight_decay=1e-4)

        for i in range(num_iters):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            out = self.model.forward(processed_image, until_layer=target_layer)
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(out[0, target_channel])
            print(f'Iteration: {i + 1}, Loss: {loss.item():.4f}')
            # Backward
            loss.backward()
            # Update image
            optimizer.step()

        return recreate_image(processed_image), processed_image

    def generate_inverted_image(self, input_image, output_size: int, target_layer: str, num_iters: int = 200):
        self._prepare_model()

        # Generate a random image which we will optimize
        opt_img = (1e-1 * torch.randn(1, 3, output_size, output_size)).requires_grad_()
        # Define optimizer for previously created image
        optimizer = SGD([opt_img], lr=1e4, momentum=0.9)
        # Get the output from the model after a forward pass until target_layer
        # with the input image (real image, NOT the randomly generated one)
        prep_img = preprocess_image(input_image)
        input_image_layer_output = self.model.forward(prep_img, until_layer=target_layer).detach()

        # Alpha regularization parameters
        # Parameter alpha, which is actually sixth norm
        alpha_reg_alpha = 6
        # The multiplier, lambda alpha
        alpha_reg_lambda = 1e-7

        # Total variation regularization parameters
        # Parameter beta, which is actually second norm
        tv_reg_beta = 2
        # The multiplier, lambda beta
        tv_reg_lambda = 1e-8

        for i in range(num_iters):
            optimizer.zero_grad()
            # Get the output from the model after a forward pass until target_layer
            # with the generated image (randomly generated one, NOT the real image)
            output = self.model.forward(opt_img, until_layer=target_layer)
            # Calculate Euclidean loss
            euc_loss = 1e-1 * normalized_euclidean_loss(input_image_layer_output, output)
            # Calculate alpha regularization
            reg_alpha = alpha_reg_lambda * alpha_norm(opt_img, alpha_reg_alpha)
            # Calculate total variation regularization
            reg_total_variation = tv_reg_lambda * total_variation_norm(opt_img, tv_reg_beta)
            # Sum all to optimize
            loss = euc_loss + reg_alpha + reg_total_variation
            # Step
            loss.backward()
            optimizer.step()
            # Generate image every 5 iterations
            if i % 5 == 0:
                print(f'Iteration: {i}, Loss: {loss.item():.4f}')
            # Reduce learning rate every 40 iterations
            if i % 40 == 0:
                for pg in optimizer.param_groups:
                    pg['lr'] *= 1 / 10

        return recreate_image(opt_img), opt_img
