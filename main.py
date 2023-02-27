import argparse
import copy
import ctypes
import gc
import json
import os
import queue
import subprocess
import sys
import threading
import time
from ctypes import *

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.ctypeslib as npCtypes
import torch
from PIL import Image
from tqdm import tqdm

from sdpc.Sdpc_struct import SqSdpcInfo
from sdpc.Sdpc import Sdpc

####################################################################################################
parser = argparse.ArgumentParser(description='Code to patch WSI using SDPC files')
parser.add_argument('--data_path', type=str,
                    default='',
                    help='path of SDPC files')

# build patches on winsdows/linux platforms
parser.add_argument('--wins_linux', type=int, default=0, choices=[0, 1], help='0:wins; 1:linux')

# auto choose gpu
parser.add_argument('--gpu_num', type=int, default=0, choices=[0, 1], help='Use gpu numbers or not')
parser.add_argument('--free_rate', type=float, default=0.8, help='use gpu with lower free rate (auto choose)')
parser.add_argument('--choose_method', type=str, default='Min', choices=['Order', 'Max', 'Min'],
                    help='auto choose gpu method')
parser.add_argument('--gpu', type=str, default='', help='specify gpu to use (ignore auto choose)')

# gerenal params.
parser.add_argument('--multiprocess_num', type=int, default=2,
                    help='the number of multiprocess for saving')
parser.add_argument('--patch_w', type=int, default=256, help='the width of patch')
parser.add_argument('--patch_h', type=int, default=256, help='the height of patch')
parser.add_argument('--overlap_w', type=int, default=0, help='the overlap width of patch')
parser.add_argument('--overlap_h', type=int, default=0, help='the overlap height of patch')

# use new/old ImageViewer software
parser.add_argument('--old_new', type=int, default=1, help='0:old version; 1:new version')
parser.add_argument('--resolution', type=float, default=0.103888,
                    help='resolution of the scanner: 0.103888um/pixel, 0.120256um/pixel, ...')
parser.add_argument('--magnification', type=int, default=80, help='magnification of the scanner: 40x, 80x, ...')
parser.add_argument('--mag_to_cut', type=float, default=5, help='magnification of the scanner: 5x, 20x, 40x, ...')

# mode 1: build patch with sdpl, mode 2: automatically build patch
parser.add_argument('--build_mode', type=int, default=1, choices=[1, 2], help='1:use sdpl; 2:no sdpl')
# mode 1 params.
parser.add_argument('--rect_color', type=str, default="Red",
                    help='color of the rectangle to cut, including "Lime", "Blue" and other rgb (e.g. "55, 213, 39")')
# mode 2 params.
parser.add_argument('--thumbnail_level', type=int, default=2, choices=[1, 2, 3],
                    help='the top level to catch thumbnail images from sdpc')
parser.add_argument('--marked_thumbnail', type=int, default=1, choices=[0, 1],
                    help='0: no produce, 1: produce marked thumbnail')
parser.add_argument('--mask', type=int, default=0, choices=[0, 1],
                    help='0: no produce, 1: produce mask')
parser.add_argument('--kernel_size', type=int, default=5, help='the kernel size of close and open opts for mask')
parser.add_argument('--blank_TH', type=float, default=0.8, help='cut patches with blank rate lower than blank_TH')


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


def get_nvidia_free_gpu(threshold=0.8, method='Order'):
    def _get_pos(command: str, start_pos: int, val: str):
        pos = command[start_pos:].find(val)
        if pos != -1:
            pos += start_pos
        return pos

    query = subprocess.getoutput('nvidia-smi -q -d Memory |grep -A4 GPU')
    free_id = []
    free_dict = {}
    gpu_id = 0
    str_scan_pos = 0

    while str_scan_pos < len(query):
        total_pos = _get_pos(query, str_scan_pos, 'Total')
        comma_pos = _get_pos(query, total_pos + 1, ':')
        _MiB_pos = _get_pos(query, comma_pos + 1, 'MiB')
        try:
            total_memory = int(query[comma_pos + 1:_MiB_pos])
        except:
            break

        _Free_pos = _get_pos(query, _MiB_pos + 1, 'Free')
        comma_pos = _get_pos(query, _Free_pos + 1, ':')
        _MiB_pos = _get_pos(query, comma_pos + 1, 'MiB')
        try:
            free_memory = int(query[comma_pos + 1:_MiB_pos])
        except:
            break
        free_rate = float(free_memory) / float(total_memory)
        if free_rate > threshold:
            free_id.append(gpu_id)
            free_dict.update({gpu_id: free_memory})
            print("GPU:%d, Free:%d, Total:%d, Free rate:%.2f." % (gpu_id, free_memory, total_memory, free_rate))
        else:
            print("GPU:%d, Free:%d, Total:%d, Free rate:%.2f, Unselected." % (
                gpu_id, free_memory, total_memory, free_rate))
        gpu_id += 1
        str_scan_pos = _MiB_pos + 1
    if method == 'Max':
        free_id = sorted([_id for _id in free_dict], key=lambda _id: free_dict[_id], reverse=True)
    elif method == 'Min':
        free_id = sorted([_id for _id in free_dict], key=lambda _id: free_dict[_id], reverse=False)
    elif method == 'Order':
        pass
    else:
        raise Exception('Wrong Choose GPU method!')
    return free_id


def config_nvidia_env(num=1, threshold=0.8, choose_method='Order', *ids):
    str_gpus = []
    if len(ids) != 0 and '' not in ids:
        if isinstance(ids, int):
            str_gpus = [str(ids)]
        elif isinstance(ids, (list, tuple)):
            for _id in ids:
                str_gpus.append(str(_id))
        else:
            raise Exception('Wrong manual gpu id type')
    else:
        if num == 0:
            str_gpus = ['-1']
        else:
            avail_gpus = get_nvidia_free_gpu(threshold, method=choose_method)
            if not avail_gpus:
                raise Exception('No free GPU with memory more than {0}%'.format(100 * threshold))
            n = 0
            for gpu in avail_gpus:
                if n >= num:
                    break
                str_gpus.append(str(gpu))
                n += 1
            if n < num:
                raise Exception('No enough free GPU with memory more than {0}%'.format(100 * threshold))
    env_val = ','.join(str_gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = env_val
    return env_val


def get_bg_mask(thumbnail, kernel_size=1):
    hsv = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2HSV)
    ret1, th1 = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_OTSU)

    close_kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    image_close = cv2.morphologyEx(np.array(th1), cv2.MORPH_CLOSE, close_kernel)
    open_kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    _image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)
    image_open = (_image_open / 255.0).astype(np.uint8)

    return image_open


def adjust_zoomscale(_args):
    _args.old_zoomscale = 0.5
    _args.new_zoomscale = _args.resolution * _args.magnification / 15
    _args.zoomscale = _args.zoomscale * _args.new_zoomscale if _args.old_new else _args.zoomscale * _args.old_zoomscale
    # judge the layer to build patches
    if _args.zoomscale <= 1 / 16:
        _args.WSI_level = 2
    elif _args.zoomscale <= 1 / 4:
        _args.WSI_level = 1
    else:
        _args.WSI_level = 0
    return _args


class MultiProcessSave(threading.Thread):
    def run(self):
        def save_img(_img_dict: dict):
            for k, v in _img_dict.items():
                v.save(k)

        event.wait(timeout=10)  # read patch with main()
        while True:  # saving patches
            # quit when q is empty and main() is stop
            if q.empty() and not event.is_set():
                break
            else:
                try:
                    img_dict = q.get(block=event.is_set(), timeout=10)  # read item
                    save_img(img_dict)  # save
                except:
                    break
        print('-----------------* Patch Saving Finished *---------------------')


# class main():  # used for debug
class main(threading.Thread):  # used for run
    def run(self):
        args = parser.parse_args()
        if args.gpu_num != 0:
            gpu_env = config_nvidia_env(args.gpu_num, args.free_rate, args.choose_method, args.gpu)
            print('Using GPU ID:', gpu_env)
            args.device = torch.device('cuda')
        else:
            args.device = torch.device('cpu')

        files = os.listdir(args.data_path)
        colors = []
        args.zoomscale = args.mag_to_cut / args.magnification
        args = adjust_zoomscale(args)

        for i, file in enumerate(files):
            if file.endswith('.sdpc'):
                # print('file = ', file)
                fi_path_, _ = file.split('.')
                fi_path = os.path.join(args.data_path, fi_path_ + '_' + str(args.mag_to_cut))
                folder = os.path.exists(fi_path)
                # 新建保存patch的文件夹
                if not folder:
                    os.makedirs(fi_path)
                else:
                    print('...There is this folder')
                    continue

                print('----------* Processing: %s *----------' % file)
                time_start = time.time()
                if args.build_mode == 1:
                    if not os.path.exists(os.path.join(args.data_path + '/' + file.replace('sdpc', 'sdpl'))):
                        continue
                    self.cut_with_sdpl(args, file, fi_path, colors)

                elif args.build_mode == 2:
                    self.auto_cut(args, file, fi_path)

                time_end = time.time()
                print('total time:', time_end - time_start)
        print('-----------------* Patch Reading Finished *---------------------')
        event.clear()

    def cut_with_sdpl(self, _args, _file, _fi_path, _colors):
        # 已经得到了每张wsi的各个patch的坐标位置
        sdpc_path = os.path.join(_args.data_path, _file)
        sdpl_path = sdpc_path.replace('.sdpc', '.sdpl')
        # 保存原有的sdpl至*_old.sdpl
        if _args.wins_linux == 1:
            os.system('cp %s %s' % (sdpl_path, sdpl_path.replace('.sdpl', '_old.sdpl')))  # Linux下拷文件
        else:
            os.system('copy %s %s' % (sdpl_path.replace('/', '\\'),
                                      sdpl_path.replace('/', '\\').replace('.sdpl', '_old.sdpl')))  # window下拷文件
        with open(sdpl_path, 'r', encoding='UTF-8') as f:
            label_dic = json.load(f)
            # 计算在第0层需要切patch的大小
            self.getcoords(_args=_args, sdpc_path=sdpc_path, label_dic=label_dic, save_dir=_fi_path,
                           patch_w=_args.patch_w, patch_h=_args.patch_h, colors=_colors, wins_linux=_args.wins_linux,
                           patch_level=_args.WSI_level, zoomscale=_args.zoomscale, rect_color=_args.rect_color)
            # 保存新的sdpl覆盖*.sdpl
            with open(sdpl_path, 'w') as f:
                json.dump(label_dic, f)

    def getcoords(self, _args, sdpc_path, label_dic, save_dir, patch_w, patch_h, colors, wins_linux=0, patch_level=0,
                  zoomscale=0.25, rect_color="Red"):
        template = None
        flag = True
        count = 0
        rec_index = None
        for counter in label_dic['LabelRoot']['LabelInfoList'][0:]:
            if counter['LabelInfo']['ToolInfor'] == "btn_brush" or counter['LabelInfo']['ToolInfor'] == "btn_pen":
                if flag:
                    template = copy.deepcopy(counter)
                    rec_index = len(label_dic['LabelRoot']['LabelInfoList']) + 1
                    template['LabelInfo']['Id'] = rec_index
                    template['LabelInfo']['PenColor'] = rect_color
                    template['LabelInfo']['FontColor'] = rect_color
                    template['LabelInfo']['ZoomScale'] = zoomscale
                    template['LabelInfo']['Dimensioning'] = 1
                    template['LabelInfo']['ToolInfor'] = "btn_rect"
                    template['PointsInfo']['ps'] = None
                    template['TextInfo']['Text'] = None
                    flag = False
                counter['LabelInfo']['Dimensioning'] = count
                count = count + 1

            if counter['LabelInfo']['PenColor'] not in colors:
                colors.append(counter['LabelInfo']['PenColor'])

        if template is None:
            print('Please add a brush annotation')
            sys.exit()
        # read sdpc
        sdpc = Sdpc(sdpc_path)
        if wins_linux == 1:
            pre_file = save_dir.split('/')[-1]
        else:
            pre_file = save_dir.replace('/', '\\').split('\\')[-1]

        for counter in tqdm(label_dic['LabelRoot']['LabelInfoList'][0:]):
            if counter['LabelInfo']['ToolInfor'] == "btn_brush" or counter['LabelInfo']['ToolInfor'] == "btn_pen":
                color_index = colors.index(counter['LabelInfo']['PenColor'])
                Points = list(zip(*[list(map(int, point.split(', '))) for point in counter['PointsInfo']['ps']]))
                Ref_x, Ref_y, _, _ = counter['LabelInfo']['CurPicRect'].split(', ')
                Ref_x, Ref_y = int(Ref_x), int(Ref_y)
                std_scale = counter['LabelInfo']['ZoomScale']
                Pointsx = []
                Pointsy = []
                for i in range(len(Points[0])):
                    Pointsx.append(int((Points[0][i] + Ref_x) * zoomscale / std_scale) - Ref_x)
                    Pointsy.append(int((Points[1][i] + Ref_y) * zoomscale / std_scale) - Ref_y)
                SPA_x, SPA_y = (min(Pointsx), min(Pointsy))
                SPB_x, SPB_y = (max(Pointsx), max(Pointsy))
                Pointslist = polygon_preprocessor([Pointsx, Pointsy])
                # *************************************************
                x0 = np.mean(np.array(Pointsx[:-1]))
                y0 = np.mean(np.array(Pointsy[:-1]))
                start_kx = -np.ceil((x0 - SPA_x - patch_w / 2) / patch_w)
                end_kx = np.ceil((SPB_x - x0 - patch_w / 2) / patch_w)
                start_ky = -np.ceil((y0 - SPA_y - patch_h / 2) / patch_h)
                end_ky = np.ceil((SPB_y - y0 - patch_h / 2) / patch_h)

                for x in range(int(start_kx), int(end_kx) + 1):
                    for y in range(int(start_ky), int(end_ky) + 1):
                        test_x = (x0 + x * patch_w)
                        test_y = (y0 + y * patch_h)

                        test_x_left = (x0 + x * patch_w - patch_w / 2)
                        test_y_bottom = (y0 + y * patch_h - patch_h / 2)

                        test_x_right = (x0 + x * patch_w + patch_w / 2)
                        test_y_top = (y0 + y * patch_h + patch_h / 2)

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

                        rect_x = test_x - patch_w / 2
                        rect_y = test_y - patch_h / 2
                        template1 = copy.deepcopy(template)
                        template1['LabelInfo']['StartPoint'] = '%d, %d' % (50, 50)
                        template1['LabelInfo']['EndPoint'] = '%d, %d' % (50 + patch_w, 50 + patch_h)
                        template1['LabelInfo']['Rect'] = '%d, %d, %d, %d' % (50, 50, patch_w, patch_h)
                        template1['LabelInfo']['CurPicRect'] = '%d, %d, %d, %d' % (
                            Ref_x + rect_x - 50, Ref_y + rect_y - 50, 0, 0)
                        try:
                            x_offset = int(patch_w / zoomscale / pow(4, patch_level))
                            y_offset = int(patch_h / zoomscale / pow(4, patch_level))
                            img = sdpc.read_region(
                                (int((Ref_x + rect_x) / zoomscale), int((Ref_y + rect_y) / zoomscale)),
                                patch_level, (x_offset, y_offset))
                        except:
                            print('(%d, %d) is out of the WSI, continue...' % (
                                (Ref_x + rect_x) * 2, (Ref_y + rect_y) * 2))
                            continue
                        save_dir1 = os.path.join(save_dir,
                                                 colors[color_index] + "_" + str(counter['LabelInfo']['Dimensioning']))
                        os.makedirs(save_dir1, exist_ok=True)
                        image_file = pre_file + '_%d.png' % template1['LabelInfo']['Id']
                        save_path = os.path.join(save_dir1, image_file)
                        im = process_data(_args, img)
                        q.put({save_path: im})  # for run
                        event.set()  # for run
                        # im.save(save_path)  # for debug

                        # add to LabelList中
                        label_dic['LabelRoot']['LabelInfoList'].append(template1)
                        # update template
                        rec_index += 1
                        template['LabelInfo']['Id'] = rec_index
                        template['LabelInfo']['Dimensioning'] = template['LabelInfo']['Dimensioning'] + 1
        sdpc.close()

    def auto_cut(self, _args, _file, fi_path):
        _sdpc_path = os.path.join(_args.data_path, _file)
        slide = Sdpc(_sdpc_path)

        _thumbnail_level = slide.level_count - _args.thumbnail_level
        _thumbnail = np.array(slide.read_region((0, 0), _thumbnail_level, slide.level_dimensions[_thumbnail_level]))
        thumbnail = cv2.cvtColor(_thumbnail, cv2.COLOR_BGRA2BGR)
        bg_mask = get_bg_mask(thumbnail, kernel_size=_args.kernel_size)
        if _args.mask == 1:
            os.makedirs(os.path.join(fi_path, 'thumbnail'), exist_ok=True)
            plt.imshow(bg_mask)
            f = plt.gcf()
            f.savefig(os.path.join(fi_path, 'thumbnail', 'mask.png'))
            f.clear()

        marked_img = thumbnail.copy()

        x_size = int(_args.patch_w / _args.zoomscale)
        y_size = int(_args.patch_h / _args.zoomscale)
        x_overlap = int(_args.overlap_w / _args.zoomscale)
        y_overlap = int(_args.overlap_h / _args.zoomscale)
        img_x, img_y = slide.level_dimensions[0]
        total_num = int(np.floor((img_x - x_size) / (x_size - x_overlap) + 1)) * \
                    int(np.floor((img_y - y_size) / (y_size - y_overlap) + 1))

        with tqdm(total=total_num) as pbar:
            for i in range(int(np.floor((img_x - x_size) / (x_size - x_overlap) + 1))):
                for j in range(int(np.floor((img_y - y_size) / (y_size - y_overlap) + 1))):
                    img_start_x = int(np.floor(i * (x_size - x_overlap) / img_x * bg_mask.shape[1]))
                    img_start_y = int(np.floor(j * (y_size - y_overlap) / img_y * bg_mask.shape[0]))
                    img_end_x = int(np.ceil((i * (x_size - x_overlap) + x_size) / img_x * bg_mask.shape[1]))
                    img_end_y = int(np.ceil((j * (y_size - y_overlap) + y_size) / img_y * bg_mask.shape[0]))
                    mask = bg_mask[img_start_y:img_end_y, img_start_x:img_end_x]

                    if np.sum(mask == 0) / mask.size < _args.blank_TH:
                        cv2.rectangle(marked_img, (img_start_x, img_start_y), (img_end_x, img_end_y), (255, 0, 0), 2)
                        x_start = int(i * (x_size - x_overlap))
                        y_start = int(j * (y_size - y_overlap))
                        x_offset = int(x_size / pow(4, _args.WSI_level))
                        y_offset = int(y_size / pow(4, _args.WSI_level))
                        img = slide.read_region((x_start, y_start), _args.WSI_level, (x_offset, y_offset))
                        save_path = os.path.join(fi_path, str(i) + '_' + str(j) + '.png')
                        im = process_data(_args, img)
                        q.put({save_path: im})  # for run
                        event.set()  # for run
                        # im.save(save_path)  # for debug

                    pbar.update(1)
        if _args.marked_thumbnail == 1:
            os.makedirs(os.path.join(fi_path, 'thumbnail'), exist_ok=True)
            Image.fromarray(marked_img).save(os.path.join(fi_path, 'thumbnail', 'thumbnail.png'))
        slide.close()


def process_data(_args, data: np.array):
    """
    Add your code here.
    """
    # DEVICES = _args.device
    processed_data = Image.fromarray(data)
    processed_data.thumbnail((_args.patch_w, _args.patch_h))
    return processed_data


if __name__ == '__main__':
    """
    Code for multi-process run
    """
    q = queue.Queue(-1)
    event = threading.Event()
    threads = [main()]
    Process_num = parser.parse_args().multiprocess_num
    for i in range(Process_num):
        threads.append(MultiProcessSave())
    for t in threads:
        t.start()

    """
    Code for debug
    Change class main(threading.Thread) into class main()，
    Comment on the code of the data preservation part in getcoords() and auto_cut().
    """
    # t = main()
    # t.run()
