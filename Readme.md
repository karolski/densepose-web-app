### Local setup
```bash
virtualenv -p python3 venv
source venv/bin/activate
./setup.sh
```
### Run application
```bash
flask run --host=0.0.0.0
```

### Test the apply_net_cpu
```bash
 python apply_net_cpu.py dump_cpu detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml model_final_162be9.pkl image.jpg --output image_densepose_contour.pkl -v
```