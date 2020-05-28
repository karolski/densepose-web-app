from typing import ClassVar
import sys
sys.path.append("./detectron2/projects/DensePose")
import pickle

from apply_net import main, DumpAction, register_action
from densepose.data.structures import DensePoseResult


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


if __name__ == "__main__":
    main()