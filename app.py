#!/usr/bin/python
# coding: utf-8
from flask import Flask, request, send_file, jsonify

from apply_net_cpu import DumpWithCpu

from keypoints_config import keypoints_config

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
    response = jsonify({
        "bbox_xywh": bbox_xywh.tolist(),
        "iuv_array": iuv_array.tolist()
    })
    return response

@app.route('/visualize', methods=['POST'])
def visualize():
    image = request.files['image']
    image.save('image.jpg')
    DumpWithCpu.external_execute()
    image_file = DumpWithCpu.visualise_results()
    image_file.seek(0)
    return send_file(image_file, attachment_filename='result.jpg')

@app.route('/find-keypoints', methods=['POST'])
def find_keypoints():
    image = request.files['image']
    image.save('image.jpg')
    DumpWithCpu.external_execute()
    keypoints = DumpWithCpu.find_keypoints_in_results(keypoints_config)
    return jsonify(keypoints)

@app.route('/visualize-keypoints', methods=['POST'])
def visualize_keypoints():
    image = request.files['image']
    image.save('image.jpg')
    DumpWithCpu.external_execute()
    image_file = DumpWithCpu.visualise_keypoints(keypoints_config)
    # image_file.seek(0)
    return send_file(image_file, attachment_filename='result.jpg')
