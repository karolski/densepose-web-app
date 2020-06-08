import io
from typing import ClassVar
import sys
sys.path.append("./detectron2/projects/DensePose")
import pickle

from apply_net import main, DumpAction, register_action
from densepose.data.structures import DensePoseResult
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2
import matplotlib.pyplot as plt
import numpy as np

@register_action
class DumpWithCpu(DumpAction):
    COMMAND: ClassVar[str] = "dump_cpu"
    @classmethod
    def setup_config(
        cls: type, *args, **kwargs
    ):
        args_list = list(args)
        opts = args_list[-1]
        opts.append("MODEL.DEVICE")
        opts.append("cpu")
        args_list[-1] = opts
        cfg = super().setup_config(*tuple(args_list), **kwargs)
        return cfg

    @classmethod
    def external_execute(cls):
        class Args:
            cfg='detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml'
            input='image.jpg'
            model='model_final_162be9.pkl'
            opts=[]
            output='image_densepose_contour.pkl'
            verbosity=1

        return cls.execute(Args)

    @classmethod
    def load_results_to_np_arrays(cls):
        f = open('image_densepose_contour.pkl', 'rb')
        data = pickle.load(f)
        img_id, instance_id = 0, 0  # Look at the first image and the first detected instance
        bbox_xywh = data[img_id]['pred_boxes_XYXY'][instance_id]
        result_encoded = data[img_id]['pred_densepose'].results[instance_id]
        iuv_arr = DensePoseResult.decode_png_data(*result_encoded)
        return bbox_xywh, iuv_arr

    @classmethod
    def visualise_results(cls):
        bbox_xywh, iuv_arr = cls.load_results_to_np_arrays()
        image = cv2.imread("image.jpg")
        x, y, w, h = int(bbox_xywh[0]), int(bbox_xywh[1]), int(bbox_xywh[2]), int(bbox_xywh[3])
        crop_img = image[y:y + h, x:x + w]

        fig = plt.figure(figsize=[12, 12])
        plt.imshow(crop_img[:, :, ::-1])
        plt.contour(iuv_arr[1, :, :], 30, linewidths=1)
        plt.contour(iuv_arr[2, :, :], 30, linewidths=1)
        plt.contour(iuv_arr[0, :, :], 1000, linewidths=1)

        output = io.BytesIO()
        FigureCanvas(fig).print_jpg(output)
        output.seek(0)
        return output

    @classmethod
    def find_keypoints_in_results(cls, target_keypoints_iuvs):
        bbox_xywh, iuv_arr = cls.load_results_to_np_arrays()
        iuv_arr = iuv_arr.T
        offset_x, offset_y = int(bbox_xywh[0]), int(bbox_xywh[1])
        result_keypoints = {}
        for keypoints_name, target_iuv in target_keypoints_iuvs.items():
            min_distance_a = 1000000
            min_distance_b = 1000000
            for ix, iy in np.ndindex(iuv_arr.shape[:2]):
                i, u, v = iuv_arr[ix, iy]
                if i != target_iuv['a']['i_body_part_classification']:
                    continue
                euclidian_distance_a = np.linalg.norm(np.array([u, v]) - np.array([target_iuv['a']['u'], target_iuv['a']['v']]))
                euclidian_distance_b = np.linalg.norm(np.array[u, v]) - np.array([target_iuv['a']['u'], target_iuv['a']['v']])
                if euclidian_distance_a < min_distance_a:
                    result_keypoints[keypoints_name]['ax'] = ix + offset_x
                    result_keypoints[keypoints_name]['ay'] = iy + offset_y
                if euclidian_distance_b < min_distance_b:
                    result_keypoints[keypoints_name]['bx'] = ix + offset_x
                    result_keypoints[keypoints_name]['by'] = iy + offset_y

        return result_keypoints









if __name__ == "__main__":
    main()