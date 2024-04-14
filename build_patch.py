import os
import cv2
import math
import time
import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sdpc.Sdpc import Sdpc
from concurrent.futures import ThreadPoolExecutor

####################################################################################################
parser = argparse.ArgumentParser(description='Code to patch WSI using SDPC files')
# dir params.
parser.add_argument('--annotation', action='store_true', help='use annotation to cut or auto cut')
parser.add_argument('--annotation_dir', type=str,
                    default='',
                    help='path of annotation files')

parser.add_argument('--data_dir', type=str,
                    default='/slide',
                    help='dir of slide files')
parser.add_argument('--save_dir', type=str,
                    default='/patch/0.4um_512',
                    help='path to save patches')
parser.add_argument('--csv_path', type=str,
                    default='/data_csv/dataset.csv',
                    help='path of csv file')

# general params.
parser.add_argument('--which2cut', type=str, default="resolution", choices=["magnification", "resolution"], 
                    help='use magnification or resolution to cut patches')
parser.add_argument('--magnification', type=float, default=20, help='magnification to build patch: 5x, 20x, 40x, ...')
parser.add_argument('--resolution', type=float, default=0.4, help='resolution to build patch: 0.103795, ... um/pixel')
parser.add_argument('--patch_w', type=int, default=512, help='the width of patch')
parser.add_argument('--patch_h', type=int, default=512, help='the height of patch')
parser.add_argument('--overlap_w', type=int, default=0, help='the overlap width of patch')
parser.add_argument('--overlap_h', type=int, default=0, help='the overlap height of patch')
parser.add_argument('--use_otsu', action='store_false', help='use otsu threshold or not')
parser.add_argument('--blank_rate_th', type=float, default=0.95, help='cut patches with a lower blank rate')

# thumbnail params.
parser.add_argument('--thumbnail_level', type=int, default=2, choices=[1, 2, 3, 4],
                    help='the top level to catch thumbnail images from sdpc')
parser.add_argument('--kernel_size', type=int, default=5, help='the kernel size of close and open opts for mask')
####################################################################################################


def polygon_preprocessor(points):
    n = len(points[0])

    for i in range(n - 2):
        if points[1][i] == points[1][i + 1]:
            points[1][i + 1] += 1
    if points[1][n - 2] == points[1][n - 1]:
        if points[1][n - 2] == points[1][n - 3] - 1:
            points[1][n - 2] -= 1
        else:
            points[1][n - 2] += 1

    points[0].append(points[0][1])
    points[1].append(points[1][1])

    return points


def inorout(points, x, y):
    n = len(points[0])
    count = 0

    for i in range(n - 2):
        if points[1][i] > points[1][i + 1]:
            if y >= points[1][i] or y < points[1][i + 1]:
                continue
            if y == points[1][i + 1] and y < points[1][i + 2]:
                continue
        else:
            if y <= points[1][i] or y > points[1][i + 1]:
                continue
            if y == points[1][i + 1] and y > points[1][i + 2]:
                continue

        if points[0][i] == points[0][i + 1]:
            if x < points[0][i]:
                count += 1

        else:
            slope = (points[1][i + 1] - points[1][i]) / (points[0][i + 1] - points[0][i])
            y_hat = slope * (x - points[0][i]) + points[1][i]
            if slope * (y - y_hat) > 0:
                count += 1

    return count


def get_bg_mask(thumbnail, kernel_size=1):
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


def isNullPatch(patch, rgbThresh=20, null_rate=0.8):
    r, g, b = patch.split()
    r_arr = np.array(r, dtype=int)
    g_arr = np.array(g, dtype=int)
    b_arr = np.array(b, dtype=int)
    rgb_arr = (np.abs(r_arr - g_arr) + np.abs(r_arr - b_arr) + np.abs(g_arr - b_arr)) / 3
    rgb_sum = np.sum(rgb_arr < rgbThresh) / patch.size[0] / patch.size[1]
    return True if rgb_sum > null_rate else False


def mag_transfer(sdpc, sdpc_path, magnification, resolution, patch_w, patch_h, 
                 overlap_w, overlap_h, which2cut="magnification"):
    if which2cut == "magnification":
        scan_mag = sdpc.readSdpc(sdpc_path).contents.picHead.contents.rate
        zoomscale = magnification / scan_mag
    else:
        scan_res = sdpc.readSdpc(sdpc_path).contents.picHead.contents.ruler
        zoomscale = scan_res / resolution
    scan_mag_step = sdpc.level_downsamples[1] / sdpc.level_downsamples[0]
    WSI_level = math.floor(math.log(1 / zoomscale, scan_mag_step))
    zoomrate = sdpc.level_downsamples[WSI_level]
    x_size = int(patch_w / zoomscale)
    y_size = int(patch_h / zoomscale)
    x_overlap = int(overlap_w / zoomscale)
    y_overlap = int(overlap_h / zoomscale)
    
    x_step, y_step = x_size - x_overlap, y_size - y_overlap
    x_offset = int(x_size / zoomrate)
    y_offset = int(y_size / zoomrate)
    return x_size, y_size, x_step, y_step, x_offset, y_offset, WSI_level


def img_detect(save_dir, slide, coord, bg_mask, marked_img, WSI_level, slide_x, slide_y, patch_w, patch_h, 
               x_size, y_size, x_offset, y_offset, use_otsu, blank_rate_th):
    x_start, y_start = coord[0], coord[1]
    mask_start_x = int(np.floor(x_start / slide_x * bg_mask.shape[1]))
    mask_start_y = int(np.floor(y_start / slide_y * bg_mask.shape[0]))
    mask_end_x = int(np.ceil((x_start + x_size) / slide_x * bg_mask.shape[1]))
    mask_end_y = int(np.ceil((y_start + y_size) / slide_y * bg_mask.shape[0]))
    mask = bg_mask[mask_start_y:mask_end_y, mask_start_x:mask_end_x]
    patch_save_path = os.path.join(save_dir, '{}_{}_{}_{}.png'.format(x_start, y_start, x_start + x_size, y_start + y_size))

    if not use_otsu or (use_otsu and (np.sum(mask == 0) / mask.size) < blank_rate_th):
        try:    
            img = slide.read_region((x_start, y_start), WSI_level, (x_offset, y_offset))
            img = Image.fromarray(img).convert('RGB')
            img.thumbnail((patch_w, patch_h))
            if not isWhitePatch(img) and not isNullPatch(img):
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
        self.kernel_size = args.kernel_size
        self.thumbnail_level = args.thumbnail_level

        self.use_otsu = args.use_otsu
        self.blank_rate_th = args.blank_rate_th

    def save_patch(self, data_path, coords=None):
        slide = Sdpc(data_path)
        x_size, y_size, x_step, y_step, x_offset, y_offset, WSI_level = mag_transfer(slide,
                                                                                     data_path,
                                                                                     self.magnification, 
                                                                                     self.resolution, 
                                                                                     self.patch_w, 
                                                                                     self.patch_h,
                                                                                     self.overlap_w,
                                                                                     self.overlap_h,
                                                                                     self.which2cut)
        _thumbnail_level = slide.level_count - self.thumbnail_level
        _thumbnail = np.array(slide.read_region((0, 0), _thumbnail_level, slide.level_dimensions[_thumbnail_level]))
        thumbnail = cv2.cvtColor(_thumbnail, cv2.COLOR_BGRA2BGR)
        bg_mask = get_bg_mask(thumbnail, kernel_size=self.kernel_size)
        marked_img = thumbnail.copy()

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
                            x_size, y_size, x_offset, y_offset, self.use_otsu, self.blank_rate_th)
                pbar.update(1)
        pool.shutdown()
        Image.fromarray(marked_img).save(os.path.join(thumbnail_save_dir, 'thumbnail.png'))
    
    def cut_with_annotation(self, _args, sdpc_path):
        json_name = os.path.basename(sdpc_path).replace('.sdpc', '.json')
        annotation_path = os.path.join(_args.annotation_dir, json_name)
        if os.path.exists(annotation_path):
            print("processing {}!".format(annotation_path))
            with open(annotation_path, 'r', encoding='UTF-8') as f:
                label_dic = json.load(f)
                coords = self.getcoords(sdpc_path, label_dic=label_dic)
                self.save_patch(sdpc_path, coords)
        else:
            print("{} does not exist!".format(annotation_path))
            pass

    def getcoords(self, data_path, label_dic):
        slide = Sdpc(data_path)
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
        for counter in label_dic['GroupModel']['Labels']:
            if counter['Type'] == "btn_brush" or counter['Type'] == "btn_pen":
                Pointsx = [int(point.get('X')) for point in counter['Coordinates']]
                Pointsy = [int(point.get('Y')) for point in counter['Coordinates']]
                SPA_x, SPA_y = (min(Pointsx), min(Pointsy))
                SPB_x, SPB_y = (max(Pointsx), max(Pointsy))
                Pointslist = polygon_preprocessor([Pointsx, Pointsy])
                # *************************************************
                x0 = np.mean(np.array(Pointsx[:-1]))
                y0 = np.mean(np.array(Pointsy[:-1]))
                start_kx = -np.ceil((x0 - SPA_x - x_size / 2) / x_step)
                end_kx = np.ceil((SPB_x - x0 - x_size / 2) / x_step)
                start_ky = -np.ceil((y0 - SPA_y - y_size / 2) / y_step)
                end_ky = np.ceil((SPB_y - y0 - y_size / 2) / y_step)

                for x in range(int(start_kx), int(end_kx) + 1):
                    for y in range(int(start_ky), int(end_ky) + 1):
                        test_x = (x0 + x * x_step)
                        test_y = (y0 + y * y_step)

                        test_x_left = (x0 + x * x_step - x_size / 2)
                        test_y_bottom = (y0 + y * x_step - y_size / 2)

                        test_x_right = (x0 + x * x_step + x_size / 2)
                        test_y_top = (y0 + y * x_step + y_size / 2)

                        count_points = []
                        count_points.append(inorout(Pointslist, test_x, test_y) % 2)  # mid
                        count_points.append(inorout(Pointslist, test_x_left, test_y_top) % 2)  # left_top
                        count_points.append(inorout(Pointslist, test_x_left, test_y_bottom) % 2)  # left_bottom
                        count_points.append(inorout(Pointslist, test_x_right, test_y_top) % 2)  # right_top
                        count_points.append(inorout(Pointslist, test_x_right, test_y_bottom) % 2)  # right_bottom
                        count_points.append(inorout(Pointslist, test_x, test_y_top) % 2)  # mid_top
                        count_points.append(inorout(Pointslist, test_x, test_y_bottom) % 2)  # mid_bottom
                        count_points.append(inorout(Pointslist, test_x_left, test_y) % 2)  # left_mid
                        count_points.append(inorout(Pointslist, test_x_right, test_y) % 2)  # right_mid

                        if sum(np.array(count_points) != 0) == 0:
                            continue
                        coords.append([test_x_left, test_y_bottom])
        return coords


class main():
    def run(self):
        args = parser.parse_args()
        Auto_Build = Slide2Patch(args)

        csv_cases = None
        if args.csv_path is not None:
            csv_file = pd.read_csv(args.csv_path, encoding="gbk")
            csv_cases = csv_file["slide_id"].dropna().tolist()

        _files = []
        for root, _, files in os.walk(args.data_dir):
            for file in files:
                if file.endswith(".sdpc"):
                    if csv_cases is None:
                        _files.append(os.path.join(root, file))
                    else:
                        if file.split(".")[0] in csv_cases:
                            _files.append(os.path.join(root, file))
                            
        for i, file in enumerate(_files):
            file_name = os.path.basename(file).split('.')[0]
            save_path = os.path.join(args.save_dir, file_name)
            if os.path.exists(save_path):
                if os.path.exists(os.path.join(save_path, 'thumbnail', 'thumbnail.png')):
                    continue

            print('----------* {}/{} Processing: {} *----------'.format(i + 1, len(_files), file))
            time_start = time.time()
            if args.annotation:
                Auto_Build.cut_with_annotation(args, file)
            else:
                Auto_Build.save_patch(file)

            time_end = time.time()
            print('total time:', time_end - time_start)

        print('-----------------* Patch Reading Finished *---------------------')

if __name__ == '__main__':    
    t = main()
    t.run()