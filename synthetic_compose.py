import cv2
import numpy as np
import random
import math
from PIL import Image
import copy
from imgaug import augmenters as iaa
import glob
import os
from progressbar import progressbar
from utils import load_and_show_meta
from shutil import rmtree

seq_background = iaa.Sequential([
    iaa.Add((-10, 10)),
    iaa.Multiply((0.9, 1.1)),
    iaa.GaussianBlur(sigma=(0.0, 2.0)),
    # iaa.ChangeColorTemperature((2000, 40000)),
    iaa.LinearContrast((0.6, 1.4)),
    iaa.AddToBrightness((-10, 10))
    #iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255))
])

def resize_background(background, max_size = 1200, min_size = 720):
    # Resize keep ratio
    height, width, _ = background.shape
    ratio = width/height
    target_ratio = min_size/max_size
    if ratio < 1:
        # width < height
        if ratio > target_ratio:
            # height >> width, resize by height
            background = cv2.resize(background, (int(ratio*max_size), max_size))
        else:
            # height >. width, resize by width
            background = cv2.resize(background, (min_size, int(min_size/ratio)))
    else:
        # width > height
        if ratio > target_ratio:
            # width >> height, resize by width
            background = cv2.resize(background, (max_size, int(max_size / ratio)))
        else:
            # width >. height, resize by height
            background = cv2.resize(background, (int(ratio * min_size), min_size))
    return background

def resize_object(content, subobj_img, background, target_area):
    # Random resize object & sub-object
    # target_area = random.uniform(min_area, max_area)
    bg_height, bg_width, _ = background.shape
    # Calculate width & height of content
    tmp = content.shape
    ct_height = tmp[0]
    ct_width  = tmp[1]

    ratio = ct_width / ct_height
    new_height = int(math.sqrt(target_area*bg_height*bg_width / ratio))
    new_width = int(new_height * ratio)
    mask_subobj = None
    if subobj_img is not None:
        # print('Resize subobj', subobj_img.shape, new_width, new_height)
        mask_subobj = cv2.resize(subobj_img, (new_width, new_height))
    return cv2.resize(content, (new_width, new_height)), mask_subobj, target_area

def fit_object(mask):
    '''
        Fit co-ordinate to object
    '''
    mask = mask[:, :, 3]
    # cv2.imwrite('mask.jpg', mask)
    height, width = mask.shape
    sum_x = np.sum(mask, axis = 0)
    x1 = 0
    for i, vl in enumerate(sum_x):
        if vl > 255:
            x1 = i
            break
    x2 = width - 1
    for i in range(len(sum_x)):
        if sum_x[width - 1 - i] > 255:
            x2 = width - 1 - i
            break
    sum_y = np.sum(mask, axis = 1)
    y1 = 0
    for i, vl in enumerate(sum_y):
        if vl > 255:
            y1 = i
            break
    y2 = height - 1
    for i in range(len(sum_y)):
        if sum_y[height - 1 - i] > 255:
            y2 = height - 1 - i
            break
    return x1, y1, x2, y2

def generate_1_image(obj_img, subobj_img, bg, opacity, target_area):
    # rubic = [86, 0, 207, 135]

    background = resize_background(bg)
    background = seq_background(image=background)

    content, mask_subobj, target_area = resize_object(obj_img, subobj_img, background, target_area)
    mask_content = copy.deepcopy(content)
    content = seq_background(image= content[:, :, :3])
    im_height, im_width, _ = background.shape
    # obj_height, obj_width, _ = content.shape

    background = Image.fromarray(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
    
    # print('1', content.shape, mask_content.shape)

    #PerspectiveTransform
    a = random.uniform(0, 0.15)
    state = random.randint(0, 120)
    per_trans = iaa.PerspectiveTransform(scale=a, keep_size=True, fit_output=True, mode="constant", seed= state)
    per_trans2 = iaa.PerspectiveTransform(scale=a, keep_size=True, fit_output=True, mode="constant", seed= state)
    per_trans3 = iaa.PerspectiveTransform(scale=a, keep_size=True, fit_output=True, mode="constant", seed= state)

    resize_content = per_trans(image = content)
    mask_content = per_trans2(image = mask_content)
    if mask_subobj is not None:
        mask_subobj = per_trans3(image = mask_subobj)
    # print('2', resize_content.shape, mask_content.shape)

    #Random Rotate
    a = random.uniform(-15, 15)
    random_rotate = iaa.Rotate(rotate=a, fit_output=True)
    resize_content = random_rotate(image= resize_content)
    mask_content = random_rotate(image= mask_content)
    if mask_subobj is not None:
        mask_subobj = random_rotate(image= mask_subobj)
    
    x1, y1, x2, y2 = fit_object(mask_content)
    if mask_subobj is not None:
        rx1, ry1, rx2, ry2 = fit_object(mask_subobj)
        rx1 -= x1
        rx2 -= x1
        ry1 -= y1
        ry2 -= y1

    
    # mask_content_vis = copy.deepcopy(mask_content[:, :, 3])
    # subobj_img_vis = copy.deepcopy(mask_subobj[:, :, 3])
    # cv2.rectangle(mask_content_vis, (x1, y1), (x2, y2), (255, 255, 255), 2)
    # cv2.rectangle(subobj_img_vis, (rx1, ry1), (rx2, ry2), (255, 255, 255), 2)
    # cv2.imwrite('mask_content.jpg', mask_content_vis)
    # cv2.imwrite('subobj_img.jpg', subobj_img_vis)

    
    resize_content = resize_content[y1:y2, x1:x2]
    mask_content = mask_content[y1:y2, x1:x2]


    contentH, contentW = resize_content.shape[:2]

    resize_content = Image.fromarray(cv2.cvtColor(resize_content, cv2.COLOR_BGRA2RGBA))
    # Check size again
    if im_width - contentW < 0 or im_height - contentH < 0:
        contentW = int(contentW/2)
        contentH = int(contentH/2)
        resize_content = resize_content.resize((contentW, contentH))
        mask_content   = np.array(Image.fromarray(mask_content[:, :, 3]).resize((contentW, contentH)), dtype = np.uint8)
    else:
        mask_content = mask_content[:, :, 3]

    offset_x = random.randint(0, im_width - contentW)
    offset_y = random.randint(0, im_height - contentH)
    random_mask = int(opacity/100*255) # Opacity
    mask = np.tile(np.array(mask_content.reshape((contentH, contentW, 1))/255*random_mask, dtype = np.uint8), (1, 1, 4))

    mask = Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGRA2RGBA))
    background.paste(resize_content, (offset_x, offset_y), mask.getchannel("A"))

    content_coor = [(offset_x + contentW/2)/im_width,
                    (offset_y + contentH/2)/im_height,
                    contentW/im_width,
                    contentH/im_height]
    subobj_coor = None
    if mask_subobj is not None:
        subobj_coor = [(offset_x + (rx1+rx2)/2)/im_width,
                        (offset_y + (ry1+ry2)/2)/im_height,
                        (rx2-rx1)/im_width,
                        (ry2-ry1)/im_height] 

    return np.array(background, dtype = np.uint8), content_coor, subobj_coor
        

class Generator():
    def __init__(self, args):
        self.obj_list = load_and_show_meta(args.info)
        self.obj_path = list(self.obj_list.keys())
        self.bg_list = glob.glob(args.background_dir + '/*.jpg')

        self.total_object = len(self.obj_path)
        self.total_background = len(self.bg_list)
        self.args = args

        print('[!] Total object: {}'.format(self.total_object))
        print('[!] Total background: {}'.format(self.total_background))

    def get_random_backround(self):
        # Random background
        idx_bg = random.randint(0, self.total_background-1)
        bg = cv2.imread(self.bg_list[idx_bg])
        return bg

    def get_random_obj(self):
        # Random index
        idx_obj = random.randint(0, self.total_object-1)
        # Random object
        obj_record = self.obj_list[self.obj_path[idx_obj]]
        obj_img = obj_record['logo']
        obj_id  = obj_record['id']
        subobj_img = None
        subobj_id  = None
        # if 'sublogo' in obj_record:
        #     subobj_img = obj_record['sublogo']
        #     subobj_id  = obj_record['sublogo_id']

        return obj_img, obj_id, subobj_img, subobj_id

    def generate_hard_database(self, numb = 1000, output_folder = ''):
        if os.path.exists(output_folder):
            rmtree(output_folder)
        os.mkdir(output_folder)
        current = len(glob.glob(output_folder + '/*.jpg'))
        for i in range(current, numb):
            n_obj = random.randint(self.args.min_n_obj, self.args.max_n_obj)
            lines = []
            # Get background
            im = self.get_random_backround()
            # Get multiple object
            for ii in range(n_obj):
                opacity = random.randint(self.args.min_opacity, 100) # opacity of object
                obj_img, obj_id, subobj_img, subobj_id = self.get_random_obj()
                target_area = random.uniform(self.args.min_area, self.args.max_area)
                im, obj_coor, subobj_coor = generate_1_image(obj_img, subobj_img, im, opacity, target_area)
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                ctx, cty, w, h = obj_coor
                if subobj_coor is not None:
                    rctx, rcty, rw, rh = subobj_coor
                    lines.append('{} {} {} {} {}\n'.format(subobj_id, rctx, rcty, rw, rh))
                lines.append('{} {} {} {} {}\n'.format(obj_id, ctx, cty, w, h))
            # Write image
            cv2.imwrite(output_folder + '/trashes11_{}.jpg'.format(i), im)
            print(i)
            # Write label
            with open(output_folder + '/trashes11_{}.txt'.format(i), 'w+') as f:
                for line in lines:
                    f.write(line)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Object detection data generator')
    # Data argument
    parser.add_argument('--info', type=str,
                        default='/data/disk1/hungpham/object-detection-generator/infor_trash.txt',
                        help='path to info file')
    parser.add_argument('--background_dir', type=str,
                        default='background',
                        help='path to background folder')
    parser.add_argument('--output_dir',
                        default='/data/disk1/hungpham/object-detection-generator/output',
                        help='path to output folder')
    # Augmentation argument
    parser.add_argument('--n', type=int,
                        default=1000,
                        help='Number of synthetic images')
    parser.add_argument('--min_opacity', type=int,
                        default=95,
                        help='min opacity of object while paste into background')
    parser.add_argument('--min_n_obj', type=int,
                        default=1,
                        help='min number of objects per image')
    parser.add_argument('--max_n_obj', type=int,
                        default=10,
                        help='max number of objects per image')
    parser.add_argument('--min_area', type=float,
                        default=0.0005,
                        help='min number of objects per image')
    parser.add_argument('--max_area', type=float,
                        default=0.0018,
                        help='max number of objects per image')


    args = parser.parse_args()

    gen = Generator(args)
    gen.generate_hard_database(numb = args.n, output_folder = args.output_dir)