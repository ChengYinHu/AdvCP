import math
import cv2
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch
import torchvision.models as models
import matplotlib.pylab as pyl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = models.resnet50(pretrained=True).eval().to(device)

#Classifier
def classify(dir, net):
    img = Image.open(dir)
    img = img.convert("RGB")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    img = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean,std = std)
    ])(img).to(device)

    f_image = net.forward(Variable(img[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]  # 从小到大排序, 再从后往前复制一遍，So相当于从大到小排序
    I = I[0:10]  # 挑最大的num_classes个(从0开始，到num_classes结束)
    # print(I)
    label = I[0]  # 最大的判断的分类
    confidence = f_image[35]

    return label, confidence

#Random generation of individual genotypes
def initiation(a):
    population, choromosome_length = a.shape
    # print('population, choromosome_length = ', population, choromosome_length)
    for i in range(0, population):
        for j in range(0, 4):
            a[i][j] = random.randint(0, 500)
        for j in range(4, 8):
            a[i][j] = random.randint(0, 333)
        for j in range(8, 11):
            a[i][j] = random.randint(0, 255)
        for j in range(11, 12):
            a[i][j] = random.randint(1, 4)/10
    return a

#Color film simulation
def img_color_film_effetc_digital(img, x1, x2, x3, x4, y1, y2, y3, y4, b, g, r, path):
    points = np.array([[(x1, y1), (x2, y2), (x3, y3), (x4, y4)]], np.int32)
    cv2.fillPoly(img, points, (r, g, b))

    cv2.imwrite(path, img)

#Color film intensity adjustment
def video_color_film_effect(img, cnt, I, path_adv):

    if cnt == 0:
        return img

    height, width, n = img.shape

    mask = {
        1: cv2.imread(path_adv),
    }

    mask[cnt] = cv2.resize(mask[cnt], (width, height), interpolation=cv2.INTER_CUBIC)

    new_img = cv2.addWeighted(img, (1 - I), mask[cnt], I, 0)

    return new_img

#Input the phenotype into the model
def img_color_film_effect(dir_read, b):

    img = cv2.imread(dir_read)

    # height, width, n = img.shape
    # print('height, width = ', height, width)

    x1, x2, x3, x4, y1, y2, y3, y4, r, g, b1, I = b[0][0], b[0][1], b[0][2], b[0][3], b[0][4], b[0][5], b[0][6], b[0][7], b[0][8], b[0][9], b[0][10], b[0][11]
    path_adv = 'adv.jpg'
    img_color_film_effetc_digital(img, x1, x2, x3, x4, y1, y2, y3, y4, r, g, b1, path_adv)

    cap = cv2.VideoCapture(dir_read)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    i = 1
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    videoWriter = cv2.VideoWriter(path_adv, fourcc, fps, (width, height))
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frame = video_color_film_effect(frame, i % 5, I, path_adv)
            # cv2.imshow('video', frame)
            videoWriter.write(frame)
            i += 1
            c = cv2.waitKey(1)
            if c == 27:
                break
        else:
            break



#Gets the label and confidence of the adversarial sample
def function(dir_read, net, b, tag_break):

    img_color_film_effect(dir_read, b)

    save_path = 'adv.jpg'

    img_show = Image.open(save_path)
    # plt.imshow(img_show)
    # plt.show()

    label_adv, conf = classify(save_path, net)

    if int(label_adv) != 35:
        # img_save = plt.imread(save_path)
        # name_save = 'result.jpg'
        # plt.imsave(name_save, img_save)
        tag_break = 1

    return label_adv, conf, tag_break