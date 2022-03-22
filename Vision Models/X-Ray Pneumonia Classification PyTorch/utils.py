import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import numpy as np


def show_image(image, label, denormalize=True):
    image = image.permute(1, 2, 0)
    mean = torch.FloatTensor([0.485, 0.456, 0.406])
    std = torch.FloatTensor([0.229, 0.224, 0.225])
    title = 'PNEUMONIA (1)' if label == 1 else 'NORMAL (0)'

    if denormalize:
        image = image * std + mean
        image = np.clip(image, 0, 1)
        plt.imshow(image)
        plt.title(title)
    else:
        plt.imshow(image)
        plt.title(title)
    plt.show()


def show_grid(images, labels, device, n_cols=4):
    images = [image.permute(1, 2, 0) for image in images]
    mean = torch.FloatTensor([0.485, 0.456, 0.406])
    std = torch.FloatTensor([0.229, 0.224, 0.225])

    images = [image * std + mean for image in images]
    images = [np.clip(image, 0, 1) for image in images]

    n_rows = int(np.ceil(len(images) / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 15))
    for r in range(4):
        for c in range(n_cols):
            index = r * n_cols + c
            if index >= len(images):
                break
            axs[r, c].imshow(images[index])
            title = 'PNEUMONIA (1)' if labels[index] == 1 else 'NORMAL (0)'
            axs[r, c].set_title(title)


def accuracy(y_pred, y_true):
    y_pred = F.softmax(y_pred, dim=1)
    top_p, top_class = y_pred.topk(1, dim=1)
    equals = top_class == y_true.view(*top_class.shape)
    return torch.mean(equals.type(torch.FloatTensor))


def view_classify(image, ps, label):
    class_name = ['NORMAL', 'PNEUMONIA']
    classes = np.array(class_name)

    ps = ps.cpu().data.numpy().squeeze()

    image = image.permute(1, 2, 0)
    mean = torch.FloatTensor([0.485, 0.456, 0.406])
    std = torch.FloatTensor([0.229, 0.224, 0.225])

    image = image * std + mean
    img = np.clip(image, 0, 1)

    fig, (ax1, ax2) = plt.subplots(figsize=(8, 12), ncols=2)
    ax1.imshow(img)
    ax1.set_title('Ground Truth : {}'.format(class_name[label]))
    ax1.axis('off')
    ax2.barh(classes, ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(classes)
    ax2.set_yticklabels(classes)
    ax2.set_title('Predicted Class')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()

    return None
