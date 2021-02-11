import os, selectivesearch, sys, math
import cfg
import cv2.cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt

def resize_image(in_image, new_width, new_height, out_image=None, resize_mode=cv2.INTER_CUBIC):
    '''

    :param in_image: 输入的图片
    :param new_width: resize后的新图片的宽
    :param new_height: resize后的新图片的长
    :param out_image: 保存resize后的新图片的地址
    :param resize_mode: 用于resize的cv2中的模式
    :return: resize后的新图片
    '''
    img = cv2.resize(in_image, (new_width, new_height), resize_mode)
    if out_image:
        cv2.imwrite(out_image, img)
    return img


def clip_pic(img, rect):
    '''

    :param img: 输入的图片
    :param rect: rect矩形框的4个参数
    :return: 输入的图片中相对应rect位置的部分 与 矩形框的一对对角点和长宽信息
    '''
    x, y, w, h = rect[0], rect[1], rect[2], rect[3]
    x_1 = x + w
    y_1 = y + h
    return img[y:y_1, x:x_1, :], [x, y, x_1, y_1, w, h]


def IOU(ver1, vertice2):
    '''
    用于计算两个矩形框的IOU
    :param ver1: 第一个矩形框
    :param vertice2: 第二个矩形框
    :return: 两个矩形框的IOU值
    '''
    vertice1 = [ver1[0], ver1[1], ver1[0] + ver1[2], ver1[1] + ver1[3]]
    area_inter = if_intersection(vertice1[0], vertice1[2], vertice1[1], vertice1[3], vertice2[0], vertice2[2],
                                 vertice2[1], vertice2[3])
    if area_inter:
        area_1 = ver1[2] * ver1[3]
        area_2 = vertice2[4] * vertice2[5]
        iou = float(area_inter) / (area_1 + area_2 - area_inter)
        return iou
    return False


def view_bar(message, num, total):
    '''
    进度条工具
    :param message: 在进度条前所要显示的信息
    :param num: 当前所已经处理了的对象的个数
    :param total: 要处理的对象的总的个数
    :return:
    '''
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = math.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t%d/%d' % (message + "\n", ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total,)
    sys.stdout.write(r)
    sys.stdout.flush()


def show_rect(img_path, regions, message):
    '''
    :param img_path: 要显示的原图片
    :param regions: 要在原图片上标注的矩形框的参数
    :param message: 在矩形框周围添加的信息
    :return:
    '''
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for x, y, w, h in regions:
        x, y, w, h = int(x), int(y), int(w), int(h)
        rect = cv2.rectangle(
            img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, message, (x + 20, y + 40), font, 1, (255, 0, 0), 2)
    plt.imshow(img)
    plt.show()


def image_proposal(img_path):
    '''
    输入要进行候选框提取的图片
    利用图片的各像素点的特点进行候选框的提取，由于候选框数量太多且针对不同的问题背景所需要的候选框的尺寸是不一样的
    因此要经过一系列的规则加以限制来进一步减小特征框的数量
    '''
    img = cv2.imread(img_path)
    img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=10)
    candidates = set()
    images = []
    vertices = []
    for r in regions:
        if r['rect'] in candidates:
            continue
        if r['size'] < 220:
            continue
        if (r['rect'][2] * r['rect'][3]) < 500:
            continue
        proposal_img, proposal_vertice = clip_pic(img, r['rect'])
        if len(proposal_img) == 0:
            continue
        x, y, w, h = r['rect']
        if w == 0 or h == 0:
            continue
        [a, b, c] = np.shape(proposal_img)
        if a == 0 or b == 0 or c == 0:
            continue
        resized_proposal_img = resize_image(proposal_img, cfg.image_size, cfg.image_size)
        candidates.add(r['rect'])
        img_float = np.asarray(resized_proposal_img, dtype="float32")
        images.append(img_float)
        vertices.append(r['rect'])
    return images, vertices


def if_intersection(xmin_a, xmax_a, ymin_a, ymax_a, xmin_b, xmax_b, ymin_b, ymax_b):
    if_intersect = False
    if xmin_a < xmax_b <= xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_a <= xmin_b < xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_b < xmax_a <= xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    elif xmin_b <= xmin_a < xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    else:
        return if_intersect
    if if_intersect:
        x_sorted_list = sorted([xmin_a, xmax_a, xmin_b, xmax_b])
        y_sorted_list = sorted([ymin_a, ymax_a, ymin_b, ymax_b])
        x_intersect_w = x_sorted_list[2] - x_sorted_list[1]
        y_intersect_h = y_sorted_list[2] - y_sorted_list[1]
        area_inter = x_intersect_w * y_intersect_h
        return area_inter


def load_2flowers_data(solver, is_svm=False, is_save=True):
    f = open(cfg.finetune_data_path)
    lines = f.readlines()
    for num, line in enumerate(lines):

        images = []
        labels = []
        labels_bbox = []
        content = line.strip().split(' ')
        image_path = content[0]

        image = cv2.imread(image_path)
        ref_rect = content[2].split(',')
        # 真实的边框位置
        ground_truth = [int(i) for i in ref_rect]
        img_lbl, regions = selectivesearch.selective_search(image, scale=500, sigma=0.9, min_size=10)
        candidate = set()
        for r in regions:
            if r['rect'] in candidate:
                continue
            if r['size'] < 200:
                continue
            if (r['rect'][2] * r['rect'][3]) < 500:
                continue
            proposal_img, proposal_vertice = clip_pic(image, r['rect'])
            if len(proposal_img) == 0:
                continue
            x, y, w, h = r['rect']
            if w == 0 or h == 0:
                continue
            [a, b, c] = np.shape(proposal_img)
            if a == 0 or b == 0 or c == 0:
                continue
            resized_proposal_img = resize_image(proposal_img, cfg.image_size, cfg.image_size)
            img_float = np.asarray(resized_proposal_img, dtype=np.float32)
            candidate.add(r['rect'])

            if is_svm:
                feature = solver.predict(np.array([img_float]))
                images.append(feature[0])
            else:
                images.append(img_float)

            iou_val = IOU(ground_truth, proposal_vertice)
            px = float(proposal_vertice[0]) + float(proposal_vertice[4] / 2.0)
            py = float(proposal_vertice[1]) + float(proposal_vertice[5] / 2.0)
            ph = float(proposal_vertice[5])
            pw = float(proposal_vertice[4])

            gx = float(ref_rect[0])
            gy = float(ref_rect[1])
            gw = float(ref_rect[2])
            gh = float(ref_rect[3])

            index = int(content[1])
            if is_svm:
                if iou_val < cfg.F_svm_threshold:
                    labels.append(0)
                else:
                    labels.append(index)

                label = np.zeros(5)
                label[1: 5] = [(gx - px) / pw, (gy - py) / ph, np.log(gw / pw), np.log(gh / ph)]
                if iou_val < cfg.F_regression_threshold:
                    label[0] = 0
                else:
                    label[0] = 1
                labels_bbox.append(label)
            else:
                label = np.zeros(cfg.F_class_num)
                # 是不是有效边框
                if iou_val < cfg.F_fineturn_threshold:
                    label[0] = 1
                else:
                    label[index] = 1
                labels.append(label)
        view_bar("Process SVM_and_Reg_image of %s" % image_path, num + 1, len(lines))
        if is_save:
            if is_svm:
                if not os.path.exists(os.path.join(cfg.SVM_and_Reg_save_path, str(content[1]))):
                    os.makedirs(os.path.join(cfg.SVM_and_Reg_save_path, str(content[1])))
                np.save((os.path.join(cfg.SVM_and_Reg_save_path, str(content[1]),
                                      content[0].split('/')[-1].split('.')[0].strip())
                         + '_data.npy'), [images, labels, labels_bbox])
            else:
                np.save((os.path.join(cfg.fineturn_save_path, content[0].split('/')[-1].split('.')[0].strip()) +
                         '_data.npy'), [images, labels])

def load_from_npy(is_svm=False):
    if is_svm:
        images, labels, rects = [], [], []
        dir_list = os.listdir(cfg.SVM_and_Reg_save_path)
        for file in dir_list:
            i, l, k = np.load(os.path.join(cfg.SVM_and_Reg_save_path, file), allow_pickle=True)
            images.extend(i)
            labels.extend(l)
            rects.extend(k)
        return images, labels, rects
    else:
        images, labels = [], []
        file_list = os.listdir(cfg.fineturn_save_path)
        for file in file_list:
            i, l = np.load(os.path.join(cfg.fineturn_save_path, file), allow_pickle=True)
            images.extend(i)
            labels.extend(l)
        return images, labels