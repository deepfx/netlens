[^gbp]: setting all negative gradients to 0 when back-propagating
[^opt]: generating an input that activates your chosen network params. This is more causal than more looking for what activates some neurons

# Netlens v. 1.4.1
`a colab of @bdurupt @cesarfm @cocokiri`
A tool to interpret *neural networks,* featuring all your favorites:

* guided back-propagation[^gbp]
* optimization[^opt]
* occlusion heatmap 
* GradCAM
* some of the above 
* Christmas Bonus: A Style Transfer module that works also with non-VGG architectures!



For more on the **pro and cons** of these techniques: [Feature Visualization](https://distill.pub/2017/feature-visualization/)

## Examples

### Attribution 

#### Backprop

[Gradient Notebook](nbs/examples/Visual-Gradient_backprop.ipynb)

**Vanilla**

![vanilla-backprop-pelican](images/readme/vanilla-backprop-pelican.png)

**Guided**

![guided-backprop-pelican](images/readme/guided-backprop-pelican.png)

**Positive and Negative Saliency**

![guided-backprop-positive-negative-saliency-pelican](images/readme/guided-backprop-positive-negative-saliency-pelican.png)

**Integrated Gradient**

![integrated-gradient-snake](images/readme/integrated-gradient-snake.png)



#### GRADCAM

[GradCam Notebook](nbs/examples/Visual-Grad_CAM.ipynb)

**Guided GRADCAM**

![guided-gradcam-relulayer4-interpolation-pelican](images/readme/guided-gradcam-relulayer4-interpolation-pelican.png)

**GRADCAM for a specific features of a layer ** *(ReLU-4)*

![gradcam-convlayer4-interpolation-pelican](images/readme/gradcam-convlayer4-interpolation-pelican.png)

![gradcam-convlayer4-no-interpolation-pelican](images/readme/gradcam-convlayer4-no-interpolation-pelican.png)

**Occlusion**

*imagine something occluded*

### Generate | Optimize for an Image 

[Visual Generation Notebook](nbs/examples/Visual-Generation.ipynb)

**DeepDreamer**

![deepDreamer](images/readme/deepDreamer.png)

**Inverted Image** (*NetDreamer)*

![generated_image](images/readme/generated_image.png)

## Install

`pip install netlens`


The standard image utils (*convert, transform, reshape*) were factored out and put into piymgy`link to pymgy`. 



## API

### Main blocks | classes

`FlatModel`

* A neural network *layer* can be sliced up in many ways:

  ![Screenshot_2019-12-04 The Building Blocks of Interpretability](/home/markus/Downloads/Screenshot_2019-12-04 The Building Blocks of Interpretability.png)

  you can view those as the semantic units of the layer / network.

* Pytorch **does not** have a nice API to access layers, channels or store their gradients (*input, weights, bias*). `FlatModel` gives a nicer wrapper that stores the forward and backward gradients in a consistent way.

`Netlens`

* accesses the preprocessed FlatModel params to display interpretations

`Hooks`

* abstraction and convenience for `Pytorch's` hook API

`NetDreamer`

* filter visualizations, generating images from fixed architecture snapshots

`Optim, Param and Renderers`

* General pipeline for `optimizing` images based on an `objective` with a given `parameterization`. Abstraction inspired by tensorflow's Lucid.

* used as specific case in `StyleTransfer`:

* ```python
  def generate_style_transfer(module: StyleTransferModule, input_img, num_steps=300, style_weight=1, content_weight=1, tv_weight=0, **kwargs):
      # create objective from the module and the weights
      objective = StyleTransferObjective(module, style_weight, content_weight, tv_weight)
      # the "parameterized" image is the image itself
      param_img = RawParam(input_img, cloned=True)
      render = OptVis(module, objective, optim=optim.LBFGS)
      thresh = (num_steps,) if isinstance(num_steps, int) else num_steps
      return render.vis(param_img, thresh, in_closure=True, callback=STCallback(), **kwargs)
  ```

`StyleTransfer` , **an Artist's Playground**

* streamlined way to run StyleTransfer  experiments, which is a specific case of image optimization
* many variables to configure (*loss functions, weighting, style and content layers that compute loss, etc.*)

`Adapter`, because there aren't pure functional standards for Deep Learning yet

* We tried to make Netlens work with multiple architectures (*not just VGG or AlexNet*). 

  Still, depending on how the architectures are implemented in the libraries, some techniques work only partially. For example, the hacky, non-functional, imperative implementation of ResNet or DenseNet in Tensorflow and also Pytorch make it hard to do attribution or guided backprop (*ReLu layers get mutated, Nested Blocks aren't pure functions, arbitrary flattening inside forward pass, etc...*).

  `adapter.py` has nursery bindings to ingest these special need cases into something the`FlatModel` class can work well with.



### The Code

Composable, elegant code is still seen as `un-pythonic`. Python is easy and productive for small, solo, interactive things and therefore the environment doesn't force users to become better programmers.  

Since it's hard to understand one-off unmaintainable, imperative, throw-away code, which unfortunately is the norm (*even in major deep learning frameworks)*, we put extra effort in making the **code clean, concise and with plenty of comments.**

Don't like it? PRs welcome. 

> Be the change you want to see in the world - Mahatma Gandhi

## Tests
You like everything breaking when you refactor or try out new things?
Exactly, so we added some tests as sanity checks. This makes it easier to tinker with the code base.

Run `pytest`or `pytest --disable-pytest-warnings` in the console.
All Tests should be in /tests folder. Imports there are `as if!` from basedirectory. Test files start with `test_...`

If you use `Pycharm`change your default test runner to pytest. 

## Prior Art

* Flashtorch 
* Distill pub for inspiration
* Lucid