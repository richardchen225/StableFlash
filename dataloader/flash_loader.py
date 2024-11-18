from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.utils.data import Dataset
import os
import cv2
import os.path
import shutil
import numpy as np


def load(objdir):

    print(f"Loading data on {objdir}.")

    img_list = []

    for file_name in ["001.png", "002.png", "normals.png", "mask.png"]:
        file_path = os.path.join(objdir, file_name)
        if os.path.isfile(file_path):
            img_list.append(file_path)

    bit_depth = 65535.0

    # loading flash/no-flash image
    img_no_flash = cv2.cvtColor(
        cv2.imread(img_list[0], flags=cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH),
        cv2.COLOR_BGR2RGB,
    )
    img_flash = cv2.cvtColor(
        cv2.imread(img_list[1], flags=cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH),
        cv2.COLOR_BGR2RGB,
    )

    # loading normal & mask
    N = cv2.cvtColor(
        cv2.imread(img_list[2], flags=cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH),
        cv2.COLOR_BGR2RGB,
    )
    mask = cv2.imread(img_list[3], flags=cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    img = img_flash
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    margin = 16
    rows, cols = np.nonzero(mask)
    r_s = 0
    r_e = 512
    c_s = 0
    c_e = 512
    if rows.size != 0 and cols.size != 0:
        rowmin = np.min(rows)
        rowmax = np.max(rows)
        row = rowmax - rowmin
        colmin = np.min(cols)
        colmax = np.max(cols)
        col = colmax - colmin
        if (
            rowmin - margin <= 0
            or rowmax + margin > img.shape[0]
            or colmin - margin <= 0
            or colmax + margin > img.shape[1]
        ):
            flag = False
        else:
            flag = True

        if row > col and flag:
            r_s = rowmin - margin
            r_e = rowmax + margin
            c_s = np.max([colmin - int(0.5 * (row - col)) - margin, 0])
            c_e = np.min([colmax + int(0.5 * (row - col)) + margin, img.shape[1]])
        elif col >= row and flag:
            r_s = np.max([rowmin - int(0.5 * (col - row)) - margin, 0])
            r_e = np.min([rowmax + int(0.5 * (col - row)) + margin, img.shape[0]])
            c_s = colmin - margin
            c_e = colmax + margin
        if flag != True:
            r_s = 0
            r_e = 512
            c_s = 0
            c_e = 512
    img_flash = img_flash[r_s:r_e, c_s:c_e, :]
    img_flash = cv2.resize(img_flash, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
    img_flash = np.float32(img_flash) / 255.0
    img_flash = 2 * img_flash - 1

    img_no_flash = img_no_flash[r_s:r_e, c_s:c_e, :]
    img_no_flash = cv2.resize(
        img_no_flash, dsize=(512, 512), interpolation=cv2.INTER_CUBIC
    )
    img_no_flash = np.float32(img_no_flash) / 255.0
    img_no_flash = 2 * img_no_flash - 1

    N = N[r_s:r_e, c_s:c_e, :]
    N = cv2.resize(N, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
    N = np.float32(N) / bit_depth
    N = 2 * N - 1
    N[:, :, 1:] *= -1.0
    N = N / (np.linalg.norm(N, ord=2, axis=2, keepdims=True) + 1e-5)

    mask = mask[r_s:r_e, c_s:c_e]
    mask = cv2.resize(mask, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
    mask = np.expand_dims(mask, axis=2)
    mask = np.float32(mask) / 255.0
    img_flash *= mask
    img_no_flash *= mask

    return img_flash, img_no_flash, N


def load_eval(objdir):

    print(f"Loading data on {objdir}.")

    img_list = []

    for file_name in ["001.png", "002.png", "mask.png"]:
        file_path = os.path.join(objdir, file_name)
        if os.path.isfile(file_path):
            img_list.append(file_path)

    bit_depth = 255.0

    # loading flash/no-flash image
    img_no_flash = cv2.cvtColor(
        cv2.imread(img_list[0], flags=cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH),
        cv2.COLOR_BGR2RGB,
    )
    img_flash = cv2.cvtColor(
        cv2.imread(img_list[1], flags=cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH),
        cv2.COLOR_BGR2RGB,
    )
    img = img_flash

    mask = cv2.imread(img_list[2], flags=cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    margin = 16
    rows, cols = np.nonzero(mask)
    rowmin = np.min(rows)
    rowmax = np.max(rows)
    row = rowmax - rowmin
    colmin = np.min(cols)
    colmax = np.max(cols)
    col = colmax - colmin
    if (
        rowmin - margin <= 0
        or rowmax + margin > img.shape[0]
        or colmin - margin <= 0
        or colmax + margin > img.shape[1]
    ):
        flag = False
    else:
        flag = True

    if row > col and flag:
        r_s = rowmin - margin
        r_e = rowmax + margin
        c_s = np.max([colmin - int(0.5 * (row - col)) - margin, 0])
        c_e = np.min([colmax + int(0.5 * (row - col)) + margin, img.shape[1]])
    elif col >= row and flag:
        r_s = np.max([rowmin - int(0.5 * (col - row)) - margin, 0])
        r_e = np.min([rowmax + int(0.5 * (col - row)) + margin, img.shape[0]])
        c_s = colmin - margin
        c_e = colmax + margin
    if flag != True:
        r_s = 0
        r_e = 512
        c_s = 0
        c_e = 512

    img_flash = img_flash[r_s:r_e, c_s:c_e, :]
    img_flash = cv2.resize(img_flash, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
    img_flash = np.float32(img_flash) / bit_depth
    img_flash = 2 * img_flash - 1

    img_no_flash = img_no_flash[r_s:r_e, c_s:c_e, :]
    img_no_flash = cv2.resize(
        img_no_flash, dsize=(512, 512), interpolation=cv2.INTER_CUBIC
    )
    img_no_flash = np.float32(img_no_flash) / 255.0
    img_no_flash = 2 * img_no_flash - 1
    mask = mask[r_s:r_e, c_s:c_e]

    mask = cv2.resize(mask, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
    mask = np.expand_dims(mask, axis=2)
    mask = np.float32(mask) / 255.0
    img_flash *= mask
    img_no_flash *= mask

    mask = np.transpose(mask, (2, 0, 1))  # 1, h, w
    return img_flash, img_no_flash, r_s, r_e, c_s, c_e


class Flash_dataset(Dataset):
    def __init__(self, args):
        root = args.traindata_dir
        root_list = []
        l = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        for i in range(10):
            root_list.append(f"{root}/flash_no_flash_{l[i]}_png")
        objlist = []
        for folder_path in root_list:
            for filename in os.listdir(folder_path):
                if (
                    os.path.exists(os.path.join(folder_path, filename, "001.png"))
                    and os.path.exists(os.path.join(folder_path, filename, "002.png"))
                    and os.path.exists(
                        os.path.join(folder_path, filename, "normals.png")
                    )
                    and os.path.exists(os.path.join(folder_path, filename, "mask.png"))
                ):
                    p = folder_path + f"/{filename}/"
                    objlist.append(p)
                else:
                    shutil.rmtree(os.path.join(folder_path, filename))
        objlist = sorted(objlist)
        self.objlist = objlist
        print(f"Found {len(self.objlist)} objects!\n")

    def __getitem__(self, index_):
        objdir = self.objlist[index_]

        i1, i2, N = load(objdir)

        img1 = i1.transpose(2, 0, 1)
        img2 = i2.transpose(2, 0, 1)
        nml = N.transpose(2, 0, 1)

        return img1, img2, nml

    def __len__(self):
        return len(self.objlist)
