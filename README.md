# vision-playground

##Milestones for Version 1.5
The repo is already better than all we have seen so far (except Lucid). I put here some ideas to really make it a home run ::
**Convenience, Convention and Smart Defaults beat Configuration**
```python
#Example:
#Infer intent>
 LayeredModule.from_nested_cnn(models.alexnet(pretrained=True))
---becomes --> 
LayeredModule(models.alexnet(pretrained=True))
#if we can't infer the model --> smart error message

#The User doesn't care about our Types behind the scences
#eg:
Netlens(somepytorchmodel, image) #should work anyways, but LOG that we converted the type implicitely

#Getting the User to AHA! Moment without friction:
Netlens(model, image).gist()
#:similar to seaborn PAIRGRID - very good
#plot quick overview (either all or a sample of layers)
#guided backprop
#occlusion
```
Batteries Included
* Include starter images for StyleTransfer and Interpret playgrounds in repo

Easy logging and error handling
* Netlens(..., Log=True) should log inbetween steps from "flattening layers", to "factorizing spatial activations" to "decorrelating X"o seaborn PAIRGRID - very good
* Map error messages * Everything that isn't NN interpretation or style transfer related should become a package or util (ala pyimgy)
to possible remedies ("X failed, did you mean....?") 

Interactive plotting
* Add Buttons to flip through layer interpretion
* Animation of the building up of semantic units (very cool) -- using anim lib for sequence of guided backprop images for example

In Search for Util. CODE KILLING!
* What are the unifying abstractions? (eg Objectives, Parametization, Optimizing, Hooking, Factorizing, Normalizing)
* Everything that isn't NN interpretation or style transfer related should become a package or util (ala pyimgy)
* I bet that a lot of code has equivalent utils somewhere in the standard lib, pytorch, numpy, itertools or wherever. It's a good exercise to search for those common utils, because people understand them. It's a common language instead of inventing our own "code (x)"

Strive for a declarative config that describes the task. Think API
* In the end  someone should be able to run `guided.py -model:/path/to/model -image /path/...` and have the output images saved in his dir
* If you achieve config level, you are language independent. People can run this with JSONs etc. . This is then API level.

Tests
* Check /tests. Functional, composable tests that run for each model, so that extending the library is easy and safe without everything breaking
* Tests are easy to write. Pytest
* run `pytest --disable-pytest-warnings -v ` in root

<img src="semantic_atoms.png"/>

## Research
* The Stanford Natural Language understanding course, just for fun
* Fastai library walk through 2.0 (jeremy) -- that's the refactoring of v1. So it's stuff we haven't seen in the library. It uses new concepts we can TAKE for this!!

##Terms
**Semantic Units**: Neurons, Spatial and Channel(detectors), Groups

## Testing
run 
`pytest`
or `pytest --disable-pytest-warnings`
all Tests should be in /tests folder. Imports there are `as if!` from basedirectory

In pycharm change your default test runner to pytest 
## Todos



## Visualization Utils

https://github.com/utkuozbulak/pytorch-cnn-visualizations
https://distill.pub/2018/building-blocks/

 
## Extra considerations
* Different similarity metrics -> Gram (autocorrelation), cross-correlation, etc.
 
##Technical
* Generic Retriever (config -> model specific traversal API)
* Mapping: names <---> NN.classes
* Read up: https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d