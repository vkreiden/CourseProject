FROM conda/miniconda3

RUN conda update conda && \
    conda install -c anaconda gensim scikit-learn && \
    conda install -c conda-forge matplotlib && \
    conda update -y smart_open && \
    \
    apt-get update && \
    apt-get install -y vim

ADD anomaly_detector.py /

ENTRYPOINT python anomaly_detector.py