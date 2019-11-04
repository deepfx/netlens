from optvis.param.color import color_mean
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


def to_valid_rgb(t, decorrelate=False, sigmoid=True):
    """Transform inner dimension of t to valid rgb colors.

    In practice this consistes of two parts:
    (1) If requested, transform the colors from a decorrelated color space to RGB.
    (2) Constrain the color channels to be in [0,1], either using a sigmoid
        function or clipping.

    Args:
      t: input tensor, innermost dimension will be interpreted as colors
        and transformed/constrained.
      decorrelate: should the input tensor's colors be interpreted as coming from
        a whitened space or not?
      sigmoid: should the colors be constrained using sigmoid (if True) or
        clipping (if False).

    Returns:
      t with the innermost dimension transformed.
    """
    if decorrelate:
        #decorrelate
        t = _linear_decorelate_color(t)
        return tf.nn.sigmoid(t)

    if decorrelate and not sigmoid:
        t += color_mean
    if sigmoid:
        return tf.nn.sigmoid(t)
    else:
        return constrain_L_inf(2 * t - 1) / 2 + 0.5


def rfft2d_freqs(h, w):
    """Computes 2D spectrum frequencies."""

    fy = np.fft.fftfreq(h)[:, None]
    # when we have an odd input dimension we need to keep one additional
    # frequency and later cut off 1 pixel
    if w % 2 == 1:
        fx = np.fft.fftfreq(w)[: w // 2 + 2]
    else:
        fx = np.fft.fftfreq(w)[: w // 2 + 1]
    return np.sqrt(fx * fx + fy * fy)



def fft_image(shape, sd=None, decay_power=1):
    """An image paramaterization using 2D Fourier coefficients."""

    sd = sd or 0.01
    batch, ch, h, w = shape
    freqs = rfft2d_freqs(h, w)

    #Todo shape!
    init_val_size = (2, batch, ch) + freqs.shape

    init_val = np.random.normal(size=init_val_size, scale=sd).astype(np.float32)
    spectrum_real_imag_t = tf.Variable(init_val)
    spectrum_t = tf.complex(spectrum_real_imag_t[0], spectrum_real_imag_t[1])

    # Scale the spectrum. First normalize energy, then scale by the square-root
    # of the number of pixels to get a unitary transformation.
    # This allows to use similar learning rates to pixel-wise optimisation.
    scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h)) ** decay_power
    scale *= np.sqrt(w * h)
    scaled_spectrum_t = scale * spectrum_t

    # convert complex scaled spectrum to shape (h, w, ch) image tensor
    # needs to transpose because irfft2d returns channels first
    image_t = tf.transpose(tf.spectral.irfft2d(scaled_spectrum_t), (0, 2, 3, 1))

    # in case of odd spatial input dimensions we need to crop
    image_t = image_t[:batch, :h, :w, :ch]
    image_t = image_t / 4.0  # TODO: is that a magic constant?
    return image_t


def pixel_image(shape, sd=None, init_val=None):
    """A naive, pixel-based image parameterization.
    Defaults to a random initialization, but can take a supplied init_val argument
    instead.

    Args:
      shape: shape of resulting image, [batch, width, height, channels].
      sd: standard deviation of param initialization noise.
      init_val: an initial value to use instead of a random initialization. Needs
        to have the same shape as the supplied shape argument.

    Returns:
      tensor with shape from first argument.
    """
    if sd is not None and init_val is not None:
        warnings.warn(
            "`pixel_image` received both an initial value and a sd argument. Ignoring sd in favor of the supplied initial value."
        )

    sd = sd or 0.01
    init_val = init_val or np.random.normal(size=shape, scale=sd).astype(np.float32)
    return tf.Variable(init_val)


def image(
    w,
    h=None,
    batch=None,
):
    h = h or w
    shape = [batch, 3, h, w]
    t = pixel_image(shape)
    # decorrelate
    return tf.nn.sigmoid(t[..., :3])

def image_f():
    #fft
    #to_rgh (inner values of decorrelated space)

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
   # param_f = lambda: param.image(80, batch=n_groups)


    #get layer
    #channel reduce
    #spatial_fac
    #channel_fac
    #sort
    #objective sum