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
    conv_id = 1
    for block_id, block in enumerate(blocks):
        if block["type"] == "convolutional":
            if int(block.get("batch_normalize", "0")) == 1:
                pass
        elif block["type"] == "":
            pass


if __name__ == '__main__':
    blocks = read_cfg("../cfg/yolov3.cfg")
    extract_weights("../weights/yolov3.weights", blocks)
