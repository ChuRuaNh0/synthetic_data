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
from math import atan2
import numpy as np
from shapely.geometry import Point, Polygon
import scipy.sparse as sps
import scipy.sparse.linalg as linalg

seq_background = iaa.Sequential([
    iaa.Add((-10, 10)),
    iaa.Multiply((0.9, 1.1)),
    iaa.GaussianBlur(sigma=(0.0, 2.0)),
    # iaa.ChangeColorTemperature((2000, 40000)),
    iaa.LinearContrast((0.6, 1.4)),
    iaa.AddToBrightness((-10, 10))
    #iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255))
])

def Random_Points_in_Polygon(polygon, number):
    points = []
    minx, miny, maxx, maxy = polygon.bounds
    while len(points) < number:
        pnt = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
        if polygon.contains(pnt):
            points.append(pnt)
    return points

def argsort(seq):
    #http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3382369#3382369
    #by unutbu
    #https://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python 
    # from Boris Gorelik
    return sorted(range(len(seq)), key=seq.__getitem__)

def rotational_sort(list_of_xy_coords, centre_of_rotation_xy_coord, clockwise=True):
    cx,cy=centre_of_rotation_xy_coord
    angles = [atan2(x-cx, y-cy) for x,y in list_of_xy_coords]
    indices = argsort(angles)
    if clockwise:
        return [list_of_xy_coords[i] for i in indices]
    else:
        return [list_of_xy_coords[i] for i in indices[::-1]]


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

def get_subimg(image, dims):
    return image[dims[0]:dims[1], dims[2]:dims[3]]

def poisson_blending(image, GRAD_MIX):
    # comparison function
    def _compare(val1, val2):
        if(abs(val1) > abs(val2)):
            return val1
        else:
            return val2
    
    # membrane (region where Poisson blending is performed)
    mask = image['mask']
    Hs,Ws = mask.shape
    num_pxls = Hs * Ws
    
    # source and target image
    source = image['source'].flatten(order='C')
    target_subimg = get_subimg(image['target'], image['dims']).flatten(order='C')

    # initialise the mask, guidance vector field and laplacian
    mask = mask.flatten(order='C')
    guidance_field = np.empty_like(mask)
    laplacian = sps.lil_matrix((num_pxls, num_pxls), dtype='float64')

    for i in range(num_pxls):
        # construct the sparse laplacian block matrix
        # and guidance field for the membrane
        if(mask[i] > 0.99):
        
            laplacian[i, i] = 4
        
        # construct laplacian, and compute source and target gradient in mask
        if(i - Ws > 0):
            laplacian[i, i-Ws] = -1
            Np_up_s = source[i] - source[i-Ws]
            Np_up_t = target_subimg[i] - target_subimg[i-Ws]
        else:
            Np_up_s = source[i]
            Np_up_t = target_subimg[i]
            
        if(i % Ws != 0):
            laplacian[i, i-1] = -1
            Np_left_s = source[i] - source[i-1]
            Np_left_t = target_subimg[i] - target_subimg[i-1]
        else:
            Np_left_s = source[i]
            Np_left_t = target_subimg[i]
            
        if(i + Ws < num_pxls):
            laplacian[i, i+Ws] = -1
            Np_down_s = source[i] - source[i+Ws]
            Np_down_t = target_subimg[i] - target_subimg[i+Ws]
        else:
            Np_down_s = source[i]
            Np_down_t = target_subimg[i]
            
        if(i % Ws != Ws-1):
            laplacian[i, i+1] = -1
            Np_right_s = source[i] - source[i+1]
            Np_right_t = target_subimg[i] - target_subimg[i+1]
        else:
            Np_right_s = source[i]
            Np_right_t = target_subimg[i]
        
        # choose stronger gradient
        if(GRAD_MIX is False):
            Np_up_t = 0
            Np_left_t = 0
            Np_down_t = 0
            Np_right_t = 0
            
            guidance_field[i] = (_compare(Np_up_s, Np_up_t) + _compare(Np_left_s, Np_left_t) + 
                            _compare(Np_down_s, Np_down_t) + _compare(Np_right_s, Np_right_t))

        else:
            # if point lies outside membrane, copy target function
            laplacian[i, i] = 1
            guidance_field[i] = target_subimg[i]
    
    return [laplacian, guidance_field]

# linear least squares solver
def linlsq_solver(A, b, dims):
    x = linalg.spsolve(A.tocsc(),b)
    return np.reshape(x,(dims[0],dims[1]))

# stitches poisson equation solution with target
def stitch_images(source, target, dims):
    target[dims[0]:dims[1], dims[2]:dims[3],:] = source
    return target

def baseline(background, resize_content, offset_x, offset_y, mask):

    background.paste(resize_content, (offset_x, offset_y), mask.getchannel("A"))

    return background

def load_image(background, resize_content,mask):
  image_data = {}
  
  # normalize the images
  image_data['source'] = cv2.normalize(background.astype('float'), None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX)
  image_data['mask'] = cv2.normalize(mask.astype('float'), None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX)
  image_data['target'] = cv2.normalize(resize_content.astype('float'), None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX)
  image_data['dims'] = [[0, 0], [1, 0], [1, 1], [0, 1]]
  return image_data

# Preprocess image
def preprocess(image_data):
    # extract image data
    source = image_data['source']
    mask = image_data['mask']
    target = image_data['target']
    
    # get image shape and offset
    Hs,Ws,_ = source.shape
    Ht,Wt,_ = target.shape
    Ho, Wo = image_data['dims']
    
    # adjust source and mask if offset is negative.
    # if mask is rolled eg. from the top it rolls 
    # to the bottom, crop the rolled portion
    if(Ho < 0):
        mask = np.roll(mask, Ho, axis=0)
        source = np.roll(source, Ho, axis=0)
        mask[Hs+Ho:,:,:] = 0 # added because Ho < 0
        source[Hs+Ho:,:,:] = 0
        Ho = 0
    if(Wo < 0):
        mask = np.roll(mask, Wo, axis=1)
        source = np.roll(source, Wo, axis=1)
        mask[:,Ws+Wo:,:] = 0
        source[:,Ws+Wo:,:] = 0
        Wo = 0
    
    # mask region on target
    H_min = Ho
    H_max = min(Ho + Hs, Ht)
    W_min = Wo
    W_max = min(Wo + Ws, Wt)
    
    # crop source and mask if they lie outside the bounds of the target
    source = source[0:min(Hs, Ht-Ho),0:min(Ws, Wt-Wo),:]
    mask = mask[0:min(Hs, Ht-Ho),0:min(Ws, Wt-Wo),:]
    
    return {'source':source, 'mask': mask, 'target': target, 'dims':[H_min,H_max,W_min,W_max]}


# performs poisson blending
def blend_image(data, BLEND_TYPE, GRAD_MIX, background, resize_content, offset_x, offset_y, mask, image_data):

    if(BLEND_TYPE == 1):
        image_solution = baseline(background, resize_content, offset_x, offset_y, mask)
        
    elif(BLEND_TYPE == 2):
        equation_param = []
        ch_data = {}
        
        # construct poisson equation 
        for ch in range(3):
            ch_data['source'] = image_data['source']
            ch_data['mask'] = image_data['mask']
            ch_data['target'] = data['target'][:,:,ch]
            ch_data['dims'] = data['dims']
            equation_param.append(poisson_blending(ch_data, GRAD_MIX))

        # solve poisson equation
        image_solution = np.empty_like(data['source'])
        for i in range(3):
            image_solution[:,:,i] = linlsq_solver(equation_param[i][0],equation_param[i][1],data['source'].shape)
        
        image_solution = stitch_images(image_solution,ch_data['target'],ch_data['dims'])
        
    else:
        # wrong option
        raise Exception('Wrong option! Available: 1. Naive, 2. Poisson')
        
    return image_solution

def generate_1_image(obj_img, subobj_img, bg, opacity, target_area, polygons, seq_bg, seq_content, option_blend):
    # rubic = [86, 0, 207, 135]
    polygons = np.array(polygons).reshape(-1,2).tolist()
    polygons = rotational_sort(polygons, polygons[0], True)

    background = resize_background(bg)

    if seq_bg:
        background = seq_background(image=background)

    content, mask_subobj, target_area = resize_object(obj_img, subobj_img, background, target_area)

    mask_content = copy.deepcopy(content)


    if seq_content:
        content = seq_background(image= content[:, :, :3])

    im_height, im_width, _ = background.shape
    # obj_height, obj_width, _ = content.shape

    background = Image.fromarray(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
    
    # print('1', content.shape, mask_content.shape)

    #PerspectiveTransform
    a = random.uniform(0, 0.01)
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
    a = random.uniform(-1, 1)
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

    try:
        resize_content = Image.fromarray(cv2.cvtColor(resize_content, cv2.COLOR_BGRA2RGBA))
    except:
        print("Error")

    # Check size again
    if im_width - contentW < 0 or im_height - contentH < 0:
        contentW = int(contentW/2)
        contentH = int(contentH/2)
        resize_content = resize_content.resize((contentW, contentH))
        mask_content   = np.array(Image.fromarray(mask_content[:, :, 3]).resize((contentW, contentH)), dtype = np.uint8)
    else:
        mask_content = mask_content[:, :, 3]

    # offset_x = random.randint(0, im_width - contentW)
    # offset_y = random.randint(0, im_height - contentH)

    # random offset paste object -> background
    coords = np.array(polygons)
    coords[:, 0] = coords[:, 0] * im_width - contentW
    coords[:, 1] = coords[:, 1] * im_height - contentH

    coords = Polygon(coords)

    while True:
        points = Random_Points_in_Polygon(coords, 1)
        print(points)

        offset_x = int(points[0].x)
        offset_y = int(points[0].y)
        print(offset_x, offset_y)

        random_mask = int(opacity/100*255) # Opacity

        mask = np.tile(np.array(mask_content.reshape((contentH, contentW, 1))/255*random_mask, dtype = np.uint8), (1, 1, 4))

        mask = Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGRA2RGBA))
        
        if option_blend == 1:
            final_image = background.paste(resize_content, (offset_x, offset_y), mask.getchannel("A"))
        
        elif option_blend == 2:
        
            # print(mask)
            # print(mask.getchannel("A"))
            # print(type(mask.getchannel("A")))
            # resize_content = np.array(resize_content)
            # print(type(resize_content.get)
            # src_mask = np.zeros(np.array(mask.getchannel("A")).shape, np.array(mask.getchannel("A")).dtype)
            # src_mask =  np.full(np.array(resize_content).shape, 255, dtype = np.uint8)
            # print(contentW, contentH)
            ctx = int(offset_x)
            cty = int(offset_y)
            center = Point(ctx, cty)
            if coords.contains(center):
                background_new = cv2.cvtColor(np.array(background), cv2.COLOR_RGB2BGR) # RGB
                resize_content_new = cv2.cvtColor(np.array(resize_content), cv2.COLOR_RGB2BGR)
                center_ = (int(center.x), int(center.y))
                print(center_)
                background_new = cv2.seamlessClone(resize_content_new, background_new, mask_content, center_, cv2.NORMAL_CLONE)
                final_image = cv2.cvtColor(background_new, cv2.COLOR_BGR2RGB)
                break
            else:
                pass
            # cv2.imwrite("1.png", mask_content)



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

    return np.array(final_image, dtype = np.uint8), content_coor, subobj_coor
        

class Generator():
    def __init__(self, args):
        self.obj_list = load_and_show_meta(args.info)
        self.obj_path = list(self.obj_list.keys())
        self.bg_list = glob.glob(args.background_dir + '/*')

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
        if 'sublogo' in obj_record:
            subobj_img = obj_record['sublogo']
            subobj_id  = obj_record['sublogo_id']

        return obj_img, obj_id, subobj_img, subobj_id

    def generate_hard_database(self, numb = 10, output_folder = ''):
        if os.path.exists(output_folder):
            rmtree(output_folder)
        os.mkdir(output_folder)
        current = len(glob.glob(output_folder + '/*.jpg'))
        print(numb)
        # print(range(current, numb))
        for i in range(current, numb):
            print(i)
            n_obj = random.randint(self.args.min_n_obj, self.args.max_n_obj)
            lines = []
            # Get background
            im = self.get_random_backround()
            # Get multiple object
            for ii in range(n_obj):
                opacity = random.randint(self.args.min_opacity, 100) # opacity of object
                obj_img, obj_id, subobj_img, subobj_id = self.get_random_obj()
                target_area = random.uniform(self.args.min_area, self.args.max_area)

                im, obj_coor, subobj_coor = generate_1_image(obj_img, subobj_img, im, opacity, target_area, self.args.polygons, self.args.seq_bg, self.args.seq_content, self.args.option_blend)
                
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                ctx, cty, w, h = obj_coor
                if subobj_coor is not None:
                    rctx, rcty, rw, rh = subobj_coor
                    if rctx < 0:
                        rctx = 0
                    if rcty < 0:
                        rcty = 0
                    lines.append('{} {} {} {} {}\n'.format(subobj_id, rctx, rcty, rw, rh))
                if ctx < 0:
                    ctx = 0
                if cty < 0:
                    cty = 0
                lines.append('{} {} {} {} {}\n'.format(obj_id, ctx, cty, w, h))
            # Write image
            cv2.imwrite(output_folder + '/test_trash_{}.jpg'.format(i), im)
            # Write label
            with open(output_folder + '/test_trash_{}.txt'.format(i), 'w+') as f:
                for line in lines:
                    f.write(line)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Object detection data generator')
    # Data argument
    parser.add_argument('--info', type=str,
                        default='info.txt',
                        help='path to info file')
    parser.add_argument('--background_dir', type=str,
                        default='background',
                        help='path to background folder')
    parser.add_argument('--output_dir',
                        default='synthetic_compose',
                        help='path to output folder')
    # Augmentation argument
    parser.add_argument('--n', type=int,
                        default=10,
                        help='Number of synthetic images')
    parser.add_argument('--min_opacity', type=int,
                        default=100,
                        help='min opacity of object while paste into background')

    parser.add_argument('--min_n_obj', type=int,
                        default=1,
                        help='min number of objects per image')

    parser.add_argument('--max_n_obj', type=int,
                        default=5,
                        help='max number of objects per image')

    parser.add_argument('--min_area', type=float,
                        default=0.003,
                        help='min number of objects per image')

    parser.add_argument('--max_area', type=float,
                        default=0.05,
                        help='max number of objects per image')

    parser.add_argument('--polygons', type=float,
                        default=[0, 0.8, 1, 0.8, 1, 1, 0, 1],
                        nargs="+",
                        help='polygons add objects')

    parser.add_argument('--seq_bg', type=bool,
                        default=False,
                        help='augmentation color background')

    parser.add_argument('--seq_content', type=bool,
                        default=False,
                        help='augmentation color object')

    parser.add_argument('--option_blend', type=int,
                        default=1,
                        help='Number of synthetic images')

    args = parser.parse_args()
    print(args.polygons)
    gen = Generator(args)
    gen.generate_hard_database(numb = args.n, output_folder = args.output_dir)
