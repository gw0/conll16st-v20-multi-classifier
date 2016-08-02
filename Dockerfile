# Docker container for conll16st-v20-multi-classifier
#
# Example:
#   DATAT=en-train DATAV=en-dev DATAX=en-trial MEM=4500M
#   DATAT=en-dev DATAV=en-trial DATAX=en-trial MEM=2500M
#   DATAT=zh-train DATAV=zh-dev DATAX=zh-trial MEM=4500M
#   DATAT=zh-dev DATAV=zh-trial DATAX=zh-trial MEM=2000M
#   NAME=conll16st-v20-21 ; echo -ne "\ek${NAME:(-6)}-$DATAT\e\\"
#   docker build -t $NAME .
#   docker run -d -m $MEM --memory-swap $MEM -v /srv/storage/conll16st:/srv/ex -e OMP_NUM_THREADS=1 --name $NAME-$DATAT $NAME ex/$NAME-$DATAT conll16st-$DATAT conll16st-$DATAV conll16st-$DATAX ex/$NAME-$DATAT --clean
#     or: -e THEANO_FLAGS='openmp=True' -e OMP_NUM_THREADS=1
#     or: -e THEANO_FLAGS='device=gpu,floatX=float32,nvcc.fastmath=True,lib.cnmem=0.7'
#   docker logs -f $NAME-$DATAT
#     less +F /srv/storage/conll16st/$NAME-$DATAT/console.log
#   docker rm -f $NAME-$DATAT
#     rm -r /srv/storage/conll16st/$NAME-$DATAT

FROM debian:jessie
MAINTAINER gw0 [http://gw.tnode.com/] <gw.2016@tnode.com>

ENV DEBIAN_FRONTEND noninteractive
WORKDIR /srv/

# packages
RUN apt-get update -qq \
 && apt-get install -y \
    python \
    python-pip \
    python-virtualenv \
    git

RUN apt-get install -y \
    g++ gfortran python-dev libopenblas-dev liblapack-dev \
    python-h5py libyaml-dev graphviz \
    pkg-config libpng-dev libfreetype6-dev

# copy datasets
ADD conll16st-en-trial/ ./conll16st-en-trial/
ADD conll16st-en-train/ ./conll16st-en-train/
ADD conll16st-en-dev/ ./conll16st-en-dev/
ADD conll16st-zh-trial/ ./conll16st-zh-trial/
ADD conll16st-zh-train/ ./conll16st-zh-train/
ADD conll16st-zh-dev/ ./conll16st-zh-dev/

# setup virtualenv
ADD requirements.sh ./
RUN ./requirements.sh

# setup parser
ADD conll16st_evaluation/ ./conll16st_evaluation/
ADD v20/ ./v20/
RUN useradd -r -d /srv parser \
 && mkdir -p /srv/ex \
 && chown -R parser:parser /srv

#XXX: patch Keras
ADD patch-models.py /srv/venv/lib/python2.7/site-packages/keras/models.py

# expose interface
VOLUME /srv/ex

USER parser
ENTRYPOINT ["/srv/venv/bin/python", "/srv/v20/run.py"]
CMD ["ex/conll16st-v20-01-trial", "conll16st-en-trial", "conll16st-en-trial", "conll16st-en-trial", "ex/conll16st-v20-01-trial"]
