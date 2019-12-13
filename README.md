# Asymptotic Risk of Bézier Simplex Fitting
This repository provides source files for reproducing the above paper submitted to AAAI2020 and experiments therein.


## Requirements
- Docker 1.13.0 or above
- Git 2.0.0 or above
- GNU Make 4.2.0 or above


## How to reproduce our results
`Dockerfile` is provided for the required software.

First, build an image on your machine.

```
$ git clone https://github.com/rafcc/aaai-20.1534.git
$ cd aaai-20.1534
$ docker build --build-arg http_proxy=$http_proxy -t .
```

Then, run a container:

```
$ docker run --rm -v $(pwd):/data -it rafcc/aaai-20.1534
```

In the container, the experimental results can be reproduced by the following commands:

```
$ cd src
$ python experiments_practical.py
$ python experiments_synthetic.py
```

Then, you will get the following directories which include experimental results:

```
../results_synthetic/
../results_practical/
```
