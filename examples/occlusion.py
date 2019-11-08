import seaborn as sns
from flashtorch.utils import ImageNetIndex
from torchvision import models

from visualization.image_proc import *
from visualization.occlusion import *


def main():
    original_img, name, target_class = get_example_data(1, img_path='../old_visual/input_images/')

    prep_img = preprocess_image(original_img)

    m_orig = models.vgg19_bn(pretrained=True)
    m_orig.eval()

    hm, cm, hm_scaled = occlusion(m_orig, prep_img, target_class)
    prob_no_occ = torch.max(hm)

    _, axes = plt.subplots(1, 3, figsize=(15, 8))

    sns.heatmap(hm, xticklabels=False, yticklabels=False, vmax=prob_no_occ, ax=axes[0])
    sns.heatmap(cm, xticklabels=False, yticklabels=False, ax=axes[1])
    axes[2].imshow(original_img)
    axes[2].imshow(hm_scaled, alpha=0.50)
    plt.show()

    det_classes, det_counts = torch.unique(cm, return_counts=True)

    ini = ImageNetIndex()
    in_labels = {v: k for k, v in ini.items()}

    print('DETECTED CLASSES')
    print('\n'.join(f'{int(cl)}\t{cnt}\t{in_labels[cl]}' for cl, cnt in zip(det_classes.tolist(), det_counts.tolist())))


if __name__ == '__main__':
    main()
