from visualization.channel_reducer import ChannelReducer
from visualization.modules import LayeredModule
from torchvision.models import vgg19, alexnet
from visualization.modules import LayeredModule
from visualization.image_proc import *

from visualization.data import get_example_data
#acts = "acts of some layer after eval()"

# We'll use ChannelReducer (a wrapper around scikit learn's factorization tools)
# to apply Non-Negative Matrix factorization (NMF).
from sklearn.decomposition import NMF


'''
1. Compare shape with colab
2. Match shapes with pyTorch implementation
3. get rid of channel reducer
3/5. Transforms and Conversions  :: make custom Types in conversion MAP???
4. print all of it:: spatial factors, channel factors
4/5. Maybe put the factors into a dict?? (wouldn't make sense because they are TRANS-LAYER!)
5. sorted vs unsorted???
6. Decorators, Decorators, Decorators!
'''

def get_spatial_factors(acts, n_group): pass

def plottable(func):
    #TODO: needs conversion map of output types -> fitting plot
    #TODO gives func another kwarg plot=True
    return func


  def _apply_flat(f, acts):
    """Utility for applying f to inner dimension of acts.

    Flattens acts into a 2D tensor, applies f, then unflattens so that all
    dimesnions except innermost are unchanged.
    """
    orig_shape = acts.shape
    acts_flat = acts.reshape([-1, acts.shape[-1]])
    new_flat = f(acts_flat)
    if not isinstance(new_flat, np.ndarray):
      return new_flat
    shape = list(orig_shape[:-1]) + [-1]
    return new_flat.reshape(shape)

def log(thing, prefix="::", show_data=False):
  print("\n \n")
  print(prefix)
  if show_data: print(thing)
  #print(thing)
  print(type(thing))
  print(getattr(thing, 'shape', "not_shape"))
  print(getattr(thing, 'size', "not_size"))

def main(): pass


if __name__ == '__main__':
    #main()
    model = alexnet(pretrained=True)
    m = LayeredModule.from_alexnet(model)
    m.eval()
    print(m)
    img, _, target = get_example_data(3, img_path='../old_visual/input_images/')
    prep_img = preprocess_image(img)
    m(prep_img)
    acts = m.hooks_layers.get_stored('features-relu-4')
    log(acts, "acts")
    n_groups = 4
    nmf = NMF(n_groups)
    flat = acts.permute((0, 2, 3, 1)).reshape((-1, acts.shape[1]))
    s = nmf.fit_transform(flat)
    spatial_factors = s.transpose(1, 0).reshape((-1,) + acts.shape[-2:]).astype("float32")
    channel_factors = nmf.components_ #components_ : array, [n_components, n_features]
                                    #Factorization matrix, sometimes called ‘dictionary’.

    x_peak = np.argmax(spatial_factors.max(1), 0) #changed to 0 bc of channel flip pytorch again
    ns_sorted = np.argsort(x_peak)

    spatial_factors = spatial_factors[ns_sorted]
    channel_factors = channel_factors[ns_sorted]
    # And create a feature visualziation of each group


    #uses 2D fft_transform
    param_f = lambda: param.image(80, batch=n_groups)


    #get layer
    #channel reduce
    #spatial_fac
    #channel_fac
    #sort
    #objective sum