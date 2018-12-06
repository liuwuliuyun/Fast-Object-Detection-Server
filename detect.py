from __future__ import division
from models import *
from utils.utils import *
from utils.datasets import *
from PIL import Image
from matplotlib.ticker import NullLocator
import time
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

Tensor = torch.cuda.FloatTensor
# Bounding-box colors
cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]
model = Darknet('config/yolov3.cfg', img_size=416)
model.load_weights('weights/yolov3.weights')
model.cuda()
classes = load_classes('config/coco.names')


def image_prepare(image_path):
    re_size = (416, 416)
    img = np.array(Image.open(image_path))
    h, w, _ = img.shape
    dim_diff = np.abs(h - w)
    # Upper (left) and lower (right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
    # Add padding
    input_img = np.pad(img, pad, 'constant', constant_values=127.5) / 255.
    # Resize and normalize
    input_img = resize(input_img, (*re_size, 3), mode='reflect')
    # Channels-first
    input_img = np.transpose(input_img, (2, 0, 1))
    # As pytorch tensor
    input_img = torch.from_numpy(input_img).float()

    input_img = Variable(input_img.type(Tensor)).unsqueeze(0)

    return input_img


def detect(input_img, img_path, filename):
    global model, classes, Tensor, cmap, colors
    prev_time = time.time()
    # Get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = non_max_suppression(detections, 80, 0.8, 0.4)[0]

    # Log progress
    current_time = time.time()
    inference_time = (current_time - prev_time)*1000
    
    # Create plot
    img = np.array(Image.open(img_path))
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # The amount of padding that was added
    pad_x = max(img.shape[0] - img.shape[1], 0) * (416 / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (416 / max(img.shape))
    # Image height and width after padding is removed
    unpad_h = 416 - pad_y
    unpad_w = 416 - pad_x
    # Draw bounding boxes and labels of detections
    if detections is not None:
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            # print('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))

            # Rescale coordinates to original dimensions
            box_h = ((y2 - y1) / unpad_h) * img.shape[0]
            box_w = ((x2 - x1) / unpad_w) * img.shape[1]
            y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
            x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            # Create a Rectangle patch
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
                                     edgecolor=color,
                                     facecolor='none')
            # Add the bbox to the plot
            ax.add_patch(bbox)
            # Add label
            plt.text(x1, y1, s=classes[int(cls_pred)], color='white', verticalalignment='top',
                     bbox={'color': color, 'pad': 0})

    # Save generated image with detections
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.savefig('./static/detections/'+filename, bbox_inches='tight', pad_inches=0.0)
    plt.close()

    return inference_time
