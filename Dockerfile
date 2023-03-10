FROM python:3.6
RUN pip install -U torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/cu101/torch_stable.html
RUN pip install cython pyyaml==5.1
RUN pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
#RUN pip install detectron2==0.1.3 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html
RUN pip install dataclasses
RUN pip install scipy
RUN pip install opencv-python
RUN pip install flask
RUN pip install av

RUN mkdir /code
COPY . /code/
WORKDIR /code

RUN git clone https://github.com/facebookresearch/detectron2
RUN pip install ./detectron2
RUN wget https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl

EXPOSE 5000
CMD flask run