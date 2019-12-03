import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import models

from netlens.data import get_example_data, IMAGENET_LABELS
from netlens.image_proc import *
from netlens.occlusion import *


def show_occlussion_example():
    original_img, name, target_class = get_example_data(1, img_path='../images/examples/')
    print(f'== EXAMPLE: OCCLUSION, image {name}, class {target_class} ==')

    prep_img = preprocess_image(original_img)

    m_orig = models.vgg19_bn(pretrained=True)
    m_orig.eval()

    hm, cm, hm_scaled = generate_occlusion_heatmap(m_orig, prep_img, target_class, verbose=True)
    prob_no_occ = torch.max(hm)

    _, axes = plt.subplots(1, 3, figsize=(15, 8))

    sns.heatmap(hm, xticklabels=False, yticklabels=False, vmax=prob_no_occ, ax=axes[0])
    axes[0].set_title('Occlusion heatmap (change in class prob)')
    sns.heatmap(cm, xticklabels=False, yticklabels=False, cmap="PiYG", ax=axes[1])
    axes[1].set_title('Predicted class vs. occlusion window')
    axes[2].imshow(original_img)
    axes[2].imshow(hm_scaled, alpha=0.50)
    axes[2].set_title('Original image')
    plt.show()

    det_classes, det_counts = torch.unique(cm, return_counts=True)

    print('DETECTED CLASSES')
    print('\n'.join(f'{int(cl)}\t{cnt}\t{IMAGENET_LABELS[cl]}' for cl, cnt in zip(det_classes.tolist(), det_counts.tolist())))


if __name__ == '__main__':
    show_occlussion_example()
