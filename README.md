## Densepose Web-App
Facebooks denspose wrapped in a flask web application serving results of denspose classifications and their visualisation. See `app.py`
Densepose: https://github.com/facebookresearch/DensePose

### Local setup
```bash
virtualenv -p python3 venv
source venv/bin/activate
./setup.sh
```
### Run application
```bash
FLASK_ENV=development flask run
```

### Test the apply_net_cpu
```bash
 python apply_net_cpu.py dump_cpu detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml model_final_162be9.pkl image.jpg --output image_densepose_contour.pkl -v
```