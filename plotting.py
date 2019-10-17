import matplotlib.pyplot as plt
from keras.utils import plot_model
import matplotlib.image as mpimg


def show_images(*imgss, figsize=(20,5), cmap='gray', title=None):
    fig, ax = plt.subplots(len(imgss), len(imgss[0]), figsize=figsize)
    for row, imgs in enumerate(imgss):
        for col, img in enumerate(imgs):
            ax[row,col].axis('off')
            ax[row,col].imshow(img, cmap=cmap)
    
    if title:
        plt.suptitle(title)

    plt.show()

    return fig, ax


def plot_model_summary(model, to_file='model.png', figsize=(20,20), dpi=150,
                       show_shapes=True, show_layer_names=True, expand_nested=True):
    plot_model(model, to_file=to_file, show_shapes=show_shapes, dpi=dpi,
               show_layer_names=show_layer_names, expand_nested=expand_nested)

    plt.figure(figsize=(20,20))
    plt.axis('off')
    plt.imshow(mpimg.imread(to_file))
    plt.show()


def plot_losses(history, skip_init=0):
    loss = history.history['loss'][skip_init:]
    val_loss = history.history['val_loss'][skip_init:]

    plt.figure()
    plt.plot(range(1,len(loss)+1), loss, label='train loss')
    plt.plot(range(1,len(loss)+1), val_loss, label='validation loss')

    plt.title('Loss')
    plt.xlabel('epochs')
    plt.ylabel('Mean Square Error')
    plt.xlim(left=0)
    plt.legend()

    plt.show()