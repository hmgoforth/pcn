export DISPLAY=:0
nvidia-docker run -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
-v /home/hgoforth/pcn:/code \
pcn:tf4-cuda8-tensorpack-py27 /bin/bash
