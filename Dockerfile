FROM ubuntu:xenial
ENV DEBIAN_FRONTEND noninteractive

ARG http_proxy

RUN apt-get update -q && apt-get install -qy \
    texlive-full \
    python-pygments gnuplot \
    make git \
    python3 python3-dev python3-pip python-setuptools\
    && cd /usr/local/bin \
    && ln -s /usr/bin/python3 python\
    && apt-get clean -y\
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*


RUN pip3 install --proxy="$http_proxy" pandas==0.23.4 numpy==1.15.1 scipy==1.1.0 scikit-learn sympy==1.1.1 typing matplotlib==2.2.3 seaborn 

WORKDIR /data
VOLUME ["/data"]
