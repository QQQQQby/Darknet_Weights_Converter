# coding: utf-8

import numpy as np


def read_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.read().strip().split("\n")]
    lines = list(filter(lambda x: len(x) > 0 and x[0] != "#", lines))
    blocks = []
    block = {}
    for line in lines:
        if line[0] == '[':
            if len(block) != 0:
                blocks.append(block)
            block = {'type': line[1:-1]}
        else:
            key_, value_ = line.split('=')
            block[key_.strip()] = value_.strip()
    blocks.append(block)
    return blocks


def extract_weights(path, blocks):
    with open(path, "rb") as f:
        header = np.fromfile(f, np.int32, 5)
        weights = np.fromfile(f, dtype=np.float32)
    ptr = 0
    in_channels = 3
    conv_id = 1
    bn_id = 1
    for block_id, block in enumerate(blocks):
        if block["type"] == "convolutional":
            out_channels = int(block["filters"])
            if int(block.get("batch_normalize", "0")) == 1:
                bn_biases = weights[ptr: ptr + out_channels].copy()
                ptr += out_channels
                bn_weights = weights[ptr: ptr + out_channels].copy()
                ptr += out_channels
                bn_running_mean = weights[ptr: ptr + out_channels].copy()
                ptr += out_channels
                bn_running_var = weights[ptr: ptr + out_channels].copy()
                ptr += out_channels
            else:
                conv_biases = weights[ptr: ptr + out_channels].copy()
                ptr += out_channels
            kernel_size = int(block["size"])
            num_weights = out_channels * in_channels * kernel_size * kernel_size

            conv_weights = weights[ptr: ptr + num_weights].copy()
            ptr += num_weights

            in_channels = out_channels

        elif block["type"] == "route":
            layers = [int(layer.strip()) for layer in block["layers"].split(",")]
            layers = [block_id + layer if layer < 0 else layer for layer in layers]
            # blocks[block_id + int(block["layers"])]
    print()


if __name__ == '__main__':
    blocks = read_cfg("../cfg/yolov3.cfg")
    extract_weights("../weights/yolov3.weights", blocks)
    # blocks = read_cfg("../cfg/yolo-fastest.cfg")
    # extract_weights("../weights/yolo-fastest.weights", blocks)
