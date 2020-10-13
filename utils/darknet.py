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
    weights_length = weights.shape[0]
    ptr = 0
    conv_id = 1
    bn_id = 1
    out_channels_list = [3]
    for block_id, block in enumerate(blocks):
        if block["type"] == "convolutional":
            print(block_id, end=" ")
            out_channels = int(block["filters"])
            in_channels = out_channels_list[-1]
            if int(block.get("batch_normalize", "0")) == 1:
                bn_biases = weights[ptr: ptr + out_channels].copy()
                ptr += out_channels
                bn_weights = weights[ptr: ptr + out_channels].copy()
                ptr += out_channels
                bn_running_mean = weights[ptr: ptr + out_channels].copy()
                ptr += out_channels
                bn_running_var = weights[ptr: ptr + out_channels].copy()
                ptr += out_channels
                # print("bn", out_channels)
            else:
                conv_biases = weights[ptr: ptr + out_channels].copy()
                ptr += out_channels
                # print("bias", out_channels)
            kernel_size = int(block["size"])
            num_weights = out_channels * in_channels * kernel_size * kernel_size

            print([out_channels, in_channels, kernel_size, kernel_size], ptr)

            conv_weights = weights[ptr: ptr + num_weights].copy()
            ptr += num_weights

            out_channels_list.append(out_channels)

        elif block["type"] == "route":
            layers = [int(layer.strip()) for layer in block["layers"].split(",")]
            out_channels = 0
            for layer in layers:
                if layer >= 0:
                    layer += 2
                out_channels += out_channels_list[layer]
            out_channels_list.append(out_channels)
        else:
            out_channels_list.append(out_channels_list[-1])

    print(ptr, weights_length)
    print(ptr - weights_length)
    assert ptr == weights_length


if __name__ == '__main__':
    blocks = read_cfg("../cfg/yolov3.cfg")
    extract_weights("../weights/yolov3.weights", blocks)
    # blocks = read_cfg("../cfg/yolov3-tiny.cfg")
    # extract_weights("../weights/yolov3-tiny.weights", blocks)
    # blocks = read_cfg("../cfg/yolov3-spp.cfg")
    # extract_weights("../weights/yolov3-spp.weights", blocks)

    # blocks = read_cfg("../cfg/yolo-fastest.cfg")
    # extract_weights("../weights/yolo-fastest.weights", blocks)
