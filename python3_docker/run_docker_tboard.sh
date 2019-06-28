export DISPLAY=:0
nvidia-docker run -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
-p 6006:6006 \
-v /home/hgoforth/pcn:/code \
pcn:tf4-cuda9-tensorpack-py35 /bin/bash
