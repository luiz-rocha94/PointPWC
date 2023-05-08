import numpy as np
import cv2


def pixel2xyz(depth, P_rect, px=None, py=None):
    assert P_rect[0,1] == 0
    assert P_rect[1,0] == 0
    assert P_rect[2,0] == 0
    assert P_rect[2,1] == 0
    assert P_rect[0,0] == P_rect[1,1]
    focal_length_pixel = P_rect[0,0]
    
    height, width = depth.shape[:2]
    if px is None:
        px = np.tile(np.arange(width, dtype=np.float32)[None, :], (height, 1))
    if py is None:
        py = np.tile(np.arange(height, dtype=np.float32)[:, None], (1, width))
    const_x = P_rect[0,2] * depth + P_rect[0,3]
    const_y = P_rect[1,2] * depth + P_rect[1,3]
    
    x = ((px * (depth + P_rect[2,3]) - const_x) / focal_length_pixel) [:, :, None]
    y = ((py * (depth + P_rect[2,3]) - const_y) / focal_length_pixel) [:, :, None]
    pc = np.concatenate((x, y, depth[:, :, None]), axis=-1)
    
    pc[..., :2] *= -1.
    return pc


def load_uint16PNG(fpath):
    px_array = cv2.imread(fpath, cv2.IMREAD_UNCHANGED).astype(np.uint16)
    if len(px_array.shape) == 3:
        px_array = px_array[:, :, [2, 1, 0]]
    return px_array


def load_disp(fpath):
    # A 0 value indicates an invalid pixel (ie, no
    # ground truth exists, or the estimation algorithm didn't produce an estimate
    # for that pixel).
    array = load_uint16PNG(fpath)
    valid = array > 0
    disp = array.astype(np.float32) / 256.0
    disp[np.logical_not(valid)] = -1.
    return disp, valid


def load_op_flow(fpath):
    array = load_uint16PNG(fpath)
    valid = array[..., -1] == 1
    array = array.astype(np.float32)
    flow = (array[..., :-1] - 2**15) / 64.
    return flow, valid


def disp_2_depth(disparity, valid_disp, FOCAL_LENGTH_PIXEL):
    BASELINE = 0.54
    depth = FOCAL_LENGTH_PIXEL * BASELINE / (disparity + 1e-5)
    depth[np.logical_not(valid_disp)] = -1.
    return depth
