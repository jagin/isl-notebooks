FROM jupyter/datascience-notebook:1fbaef522f17

LABEL Jarosław Gilewski <jgilewski@jagin.pl>

RUN conda install -y pandas==0.20.3 && \
    conda install -y -c r r-car=2.1_4