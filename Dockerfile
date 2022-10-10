FROM python:3.8

RUN apt-get update

WORKDIR "/data"

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# docker run -v ~/Documents/Work/:/data/ -p 9998:8888 -p 9997:8887 -t imagereltagger /bin/bash

# RUN pip3 --no-cache-dir install jupyter && \
#     mkdir /root/.jupyter && \
#     echo "c.NotebookApp.ip = '*'" \
#          "\nc.NotebookApp.open_browser = False" \
#          "\nc.NotebookApp.token = ''" \
#          > /root/.jupyter/jupyter_notebook_config.py
# EXPOSE 8888
# CMD jupyter notebook --allow-root --port 8888
