export DISPLAY=:0
nvidia-docker run -it --rm -p "8888:8888" -e DISPLAY=$DISPLAY -e PYTHONPATH=/code \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /home/chao/packages/bluerythem/pcn:/code \
  pcn:tf4-cuda8-tensorpack-py27 \
 jupyter notebook notebooks --port=8888 --ip=0.0.0.0 --allow-root --no-browser \
--NotebookApp.token='' --NotebookApp.password=''
