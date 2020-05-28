#!/usr/bin/python
# coding: utf-8
from flask import Flask, request
from apply_net_cpu import DumpWithCpu
import json
app = Flask(__name__)

@app.route('/')
def entry_point():
    return 'Densepose Service'

@app.route('/analyze', methods=['POST'])
def analyze():
    image = request.files['image']
    image.save('image.jpg')
    DumpWithCpu.external_execute()
    bbox_xywh, iuv_array = DumpWithCpu.load_results_to_np_arrays()
    response = json.dumps({
        "bbox_xywh": bbox_xywh.tolist(),
        "iuv_array": iuv_array.tolist()
    })
    return response