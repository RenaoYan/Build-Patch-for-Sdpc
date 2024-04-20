import os
import cv2
import math
import time
import json
import glob
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

SLIDE_FORMAT = ["sdpc", "svs", "ndpi", "tiff", "tif", "dcm", "svslide", "bif", "vms", "vmu", "mrxs", "scn"]
ANNOTATION_FORMAT = ["sdpl", "json"]

parser = argparse.ArgumentParser(description='Code to tile WSI')
# necessary params.
parser.add_argument('--data_dir', type=str,
                    default='',
                    help='dir of slide files')
parser.add_argument('--save_dir', type=str,
                    default='',
                    help='path to save patches')

# optional params.
parser.add_argument('--annotation_dir', type=str,
                    default='',
                    help='path of annotation files (optional)')
parser.add_argument('--csv_path', type=str,
                    default=None,
                    help='path of csv file (optional)')

# general params. (we set the layer at the highest magnification as 40x if no magnification is provided in slide properties)
parser.add_argument('--which2cut', type=str, default="magnification", choices=["magnification", "resolution"], 
                    help='use magnification or resolution to cut patches')
parser.add_argument('--magnification', type=float, default=20, help='magnification to build patch: 5x, 20x, 40x, ...')
parser.add_argument('--resolution', type=float, default=0.4, help='resolution to build patch: 0.103795, ... um/pixel')
parser.add_argument('--patch_w', type=int, default=256, help='the width of patch')
parser.add_argument('--patch_h', type=int, default=256, help='the height of patch')
parser.add_argument('--overlap_w', type=int, default=0, help='the overlap width of patch')
parser.add_argument('--overlap_h', type=int, default=0, help='the overlap height of patch')

parser.add_argument('--thumbnail_level', type=int, default=2, choices=[1, 2, 3, 4],
                    help='the top level to catch thumbnail images from sdpc')
parser.add_argument('--use_otsu', action='store_false', help='use otsu threshold or not')
parser.add_argument('--blank_rate_th', type=float, default=0.95, help='cut patches with a lower blank rate')
parser.add_argument('--null_th', type=int, default=10, help='the threshold to drop null patches (larger to drop more): 5, 10, 15, 20, ...')


def get_bg_mask(thumbnail, kernel_size=5):
    hsv = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2HSV)
    _, th1 = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_OTSU)

    close_kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    image_close = cv2.morphologyEx(np.array(th1), cv2.MORPH_CLOSE, close_kernel)
    open_kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    _image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)
    image_open = (_image_open / 255.0).astype(np.uint8)

    return image_open


def isWhitePatch(patch, satThresh=5):
    patch = np.array(patch)
    patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    return True if np.mean(patch_hsv[:, :, 1]) < satThresh else False


def isNullPatch(patch, rgbThresh=10, null_rate=0.9):
    r, g, b = patch.split()
    r_arr = np.array(r, dtype=int)
    g_arr = np.array(g, dtype=int)
    b_arr = np.array(b, dtype=int)
    rgb_arr = (np.abs(r_arr - g_arr) + np.abs(r_arr - b_arr) + np.abs(g_arr - b_arr)) / 3
    rgb_sum = np.sum(rgb_arr < rgbThresh) / patch.size[0] / patch.size[1]
    return True if rgb_sum > null_rate else False


def mag_transfer(slide, sdpc_path, magnification, resolution, patch_w, patch_h, 
                 overlap_w, overlap_h, which2cut="magnification"):
    if which2cut == "magnification":
        if sdpc_path.endswith(".sdpc"):
            scan_mag = slide.readSdpc(sdpc_path).contents.picHead.contents.rate
        else:
            if "aperio.AppMag" in slide.properties.keys():
                scan_mag = float(slide.properties["aperio.AppMag"])
            else:
                scan_mag = 40.0
        zoomscale = magnification / scan_mag
    else:
        if sdpc_path.endswith(".sdpc"):
            scan_res = slide.readSdpc(sdpc_path).contents.picHead.contents.ruler
        else:
            scan_res = float(slide.properties["openslide.mpp-x"])
        zoomscale = scan_res / resolution
    scan_mag_step = slide.level_downsamples[1] / slide.level_downsamples[0]
    WSI_level = math.floor(math.log(1 / zoomscale, scan_mag_step))
    zoomrate = slide.level_downsamples[WSI_level]
    x_size = int(patch_w / zoomscale)
    y_size = int(patch_h / zoomscale)
    x_overlap = int(overlap_w / zoomscale)
    y_overlap = int(overlap_h / zoomscale)
    
    x_step, y_step = x_size - x_overlap, y_size - y_overlap
    x_offset = int(x_size / zoomrate)
    y_offset = int(y_size / zoomrate)
    return x_size, y_size, x_step, y_step, x_offset, y_offset, WSI_level


def img_detect(save_dir, slide, coord, bg_mask, marked_img, WSI_level, slide_x, slide_y, patch_w, patch_h, 
               x_size, y_size, x_offset, y_offset, use_otsu, blank_rate_th, rgbThresh):
    x_start, y_start = coord[0], coord[1]
    mask_start_x = int(np.floor(x_start / slide_x * bg_mask.shape[1]))
    mask_start_y = int(np.floor(y_start / slide_y * bg_mask.shape[0]))
    mask_end_x = int(np.ceil((x_start + x_size) / slide_x * bg_mask.shape[1]))
    mask_end_y = int(np.ceil((y_start + y_size) / slide_y * bg_mask.shape[0]))
    mask = bg_mask[mask_start_y:mask_end_y, mask_start_x:mask_end_x]
    patch_save_path = os.path.join(save_dir, '{}_{}_{}_{}.png'.format(x_start, y_start, x_start + x_size, y_start + y_size))

    img_flag = False
    if not use_otsu:
        img_flag = True
    elif mask.size > 0 and (np.sum(mask == 0) / mask.size) < blank_rate_th:
        img_flag = True
    if img_flag:
        try:
            img = slide.read_region((x_start, y_start), WSI_level, (x_offset, y_offset))
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img = img.convert('RGB')
            img.thumbnail((patch_w, patch_h))
            if not isWhitePatch(img) and not isNullPatch(img, rgbThresh=rgbThresh):
                cv2.rectangle(marked_img, (mask_start_x, mask_start_y), (mask_end_x, mask_end_y), (255, 0, 0), 2)
                img.save(patch_save_path)
        except Exception as e:
            print(str(e))


class Slide2Patch():
    def __init__(self, args):
        # general params.
        self.patch_w, self.patch_h = args.patch_w, args.patch_h
        self.overlap_w, self.overlap_h = args.overlap_w, args.overlap_h
        self.which2cut = args.which2cut
        self.magnification = args.magnification
        self.resolution = args.resolution

        self.save_dir = args.save_dir
        self.null_th = args.null_th
        self.thumbnail_level = args.thumbnail_level

        self.use_otsu = args.use_otsu
        self.blank_rate_th = args.blank_rate_th

    def save_patch(self, data_path, coords=None):
        if data_path.split(".")[-1] == "sdpc":
            from sdpc.Sdpc import Sdpc
            slide = Sdpc(data_path)
            thumbnail = slide.read_region((0, 0), slide.level_count - self.thumbnail_level, slide.level_dimensions[-self.thumbnail_level])
        else:
            import openslide
            slide = openslide.open_slide(data_path)
            thumbnail = slide.get_thumbnail(slide.level_dimensions[-self.thumbnail_level])
        if isinstance(thumbnail, np.ndarray):
            pass
        else:
            thumbnail = np.array(thumbnail.convert('RGB'))
        bg_mask = get_bg_mask(thumbnail)
        marked_img = thumbnail.copy()

        x_size, y_size, x_step, y_step, x_offset, y_offset, WSI_level = mag_transfer(slide,
                                                                                     data_path,
                                                                                     self.magnification, 
                                                                                     self.resolution, 
                                                                                     self.patch_w, 
                                                                                     self.patch_h,
                                                                                     self.overlap_w,
                                                                                     self.overlap_h,
                                                                                     self.which2cut)

        slide_x, slide_y = slide.level_dimensions[0]
        thumbnail_save_dir = os.path.join(self.save_dir, os.path.basename(data_path).split('.')[0], 'thumbnail')
        os.makedirs(thumbnail_save_dir, exist_ok=True)
        save_dir = os.path.join(self.save_dir, os.path.basename(data_path).split('.')[0])

        if coords is None:
            coords = []
            for i in range(int(np.floor((slide_x - x_size) / x_step + 1))):
                for j in range(int(np.floor((slide_y - y_size) / y_step + 1))):
                    coords.append([i * x_step, j * y_step])
        pool = ThreadPoolExecutor(20)
        with tqdm(total=len(coords)) as pbar:
            for coord in coords:
                pool.submit(img_detect, save_dir, slide, coord, bg_mask, marked_img, WSI_level, slide_x, slide_y, self.patch_w, self.patch_h, 
                            x_size, y_size, x_offset, y_offset, self.use_otsu, self.blank_rate_th, self.null_th)
                pbar.update(1)
        pool.shutdown()
        Image.fromarray(marked_img).save(os.path.join(thumbnail_save_dir, 'thumbnail.png'))
    
    def cut_with_annotation(self, annotation_dir, sdpc_path):
        json_name = os.path.basename(sdpc_path).split(".")[0] + ".*"
        annotation_paths = glob.glob(os.path.join(annotation_dir, json_name))
        for annotation_path in annotation_paths:
            if annotation_path.split(".")[-1] in ANNOTATION_FORMAT:
                print("processing {}!".format(annotation_path))
                with open(annotation_path, 'r', encoding='UTF-8') as f:
                    label_dic = json.load(f)
                    coords = self.getcoords(sdpc_path, label_dic=label_dic)
                    self.save_patch(sdpc_path, coords)
                return None

        print("annotation of {} does not exist!".format(sdpc_path))
        return None

    def getcoords(self, data_path, label_dic):
        if data_path.split(".")[-1] == "sdpc":
            from sdpc.Sdpc import Sdpc
            slide = Sdpc(data_path)
        else:
            import openslide
            slide = openslide.open_slide(data_path)
        x_size, y_size, x_step, y_step, _, _, _ = mag_transfer(slide,
                                                               data_path,
                                                               self.magnification, 
                                                               self.resolution, 
                                                               self.patch_w, 
                                                               self.patch_h,
                                                               self.overlap_w,
                                                               self.overlap_h,
                                                               self.which2cut)
        coords = []
        if 'GroupModel' in label_dic.keys():
            counters = label_dic['GroupModel']['Labels']
        else:
            counters = label_dic['LabelRoot']['LabelInfoList']
        for counter in counters:
            if 'Type' in counter.keys():
                counter_type = counter['Type']
            else:
                counter_type = counter['LabelInfo']['ToolInfor']
            if counter_type == "btn_brush" or counter_type == "btn_pen":
                if 'Coordinates' in counter.keys():
                    Pointsx = [int(point.get('X')) for point in counter['Coordinates']]
                    Pointsy = [int(point.get('Y')) for point in counter['Coordinates']]
                else:
                    Points = list(zip(*[list(map(int, point.split(', '))) for point in counter['PointsInfo']['ps']]))
                    Ref_x, Ref_y, _, _ = counter['LabelInfo']['CurPicRect'].split(', ')
                    Ref_x, Ref_y = int(Ref_x), int(Ref_y)
                    std_scale = counter['LabelInfo']['ZoomScale']
                    Pointsx = []
                    Pointsy = []
                    for i in range(len(Points[0])):
                        Pointsx.append(int((Points[0][i] + Ref_x) / std_scale) - Ref_x)
                        Pointsy.append(int((Points[1][i] + Ref_y) / std_scale) - Ref_y)
                SPA_x, SPA_y = (min(Pointsx), min(Pointsy))
                SPB_x, SPB_y = (max(Pointsx), max(Pointsy))
                Pointslist = np.array([Pointsx, Pointsy]).transpose(1, 0)
                
                x0 = np.mean(np.array(Pointsx[:-1]))
                y0 = np.mean(np.array(Pointsy[:-1]))
                start_kx = -np.ceil((x0 - SPA_x - x_size / 2) / x_step)
                end_kx = np.ceil((SPB_x - x0 - x_size / 2) / x_step)
                start_ky = -np.ceil((y0 - SPA_y - y_size / 2) / y_step)
                end_ky = np.ceil((SPB_y - y0 - y_size / 2) / y_step)

                for x in range(int(start_kx), int(end_kx) + 1):
                    for y in range(int(start_ky), int(end_ky) + 1):
                        test_x_left = int(x0 + x * x_step - x_size / 2)
                        test_y_bottom = int(y0 + y * x_step - y_size / 2)
                        test_x_right = int(x0 + x * x_step + x_size / 2)
                        test_y_top = int(y0 + y * x_step + y_size / 2)
                        test_list = [(x0 + x * x_step, y0 + y * x_step),
                                     (test_x_left, test_y_top), 
                                     (test_x_left, test_y_bottom), 
                                     (test_x_right, test_y_top), 
                                     (test_x_right, test_y_bottom)]
                        for test_point in test_list:
                            if cv2.pointPolygonTest(Pointslist, test_point, False) >= 0:
                                coords.append([test_x_left, test_y_bottom])
                                continue
        return coords


if __name__ == '__main__':    
    args = parser.parse_args()
    Auto_Build = Slide2Patch(args)

    csv_cases = None
    csv_slides = None
    if args.csv_path is not None:
        csv_file = pd.read_csv(args.csv_path, encoding="gbk")
        csv_slides = csv_file["slide_id"].dropna().tolist()

    _files = []
    for root, _, files in os.walk(args.data_dir):
        for file in files:
            file_format = file.split(".")[-1]
            if file_format in SLIDE_FORMAT:
                if csv_cases is None and csv_slides is None:
                    _files.append(os.path.join(root, file))
                else:
                    file_slide_id = file.split(".")[0]
                    if file_slide_id in csv_slides:
                        _files.append(os.path.join(root, file))
    _files = sorted(_files)
    
    for i, file in enumerate(_files):
        file_name = os.path.basename(file).split('.')[0]
        save_path = os.path.join(args.save_dir, file_name)
        if os.path.exists(save_path):
            if os.path.exists(os.path.join(save_path, 'thumbnail', 'thumbnail.png')):
                continue

        print('----------* {}/{} Processing: {} *----------'.format(i + 1, len(_files), file))
        time_start = time.time()
        if os.path.exists(args.annotation_dir):
            Auto_Build.cut_with_annotation(args.annotation_dir, file)
        else:
            Auto_Build.save_patch(file)

        time_end = time.time()
        print('total time:', time_end - time_start)

    print('-----------------* Patch Reading Finished *---------------------')