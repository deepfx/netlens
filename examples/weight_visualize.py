from torchvision.models import vgg16, alexnet

from visualization.modules import LayeredModule
from visualization.weight_visualize import plot_weights

def main():

    '''

       vgg = vgg16(pretrained=True)
    m = LayeredModule.from_cnn(vgg)

    #3x3 is boring to visualize.
    #Alexnet has 11x11 ..that's interesting
    first_conv = m.layers['Sequential-0'][0]
    last_conv = m.layers['Sequential-0'][28]
    #print(m, first_conv)
    #print(first_conv.weight, first_conv.weight.data, first_conv.weight.shape)
    plot_weights(last_conv)
    :return:
    '''



    model = alexnet(pretrained=True)
    alex = LayeredModule.from_nested_cnn(model)

    print(alex, "alex")
    plot_weights(alex.layers['features-conv-0'])




if __name__ == '__main__':
    main()