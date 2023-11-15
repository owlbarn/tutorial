FROM owlbarn/book:base
USER root

# prepare dependency

RUN cd /home/opam/opam-repository \
    && git remote set-url origin https://github.com/ocaml/opam-repository \
    && git pull --quiet origin master
RUN opam install core async lambdasoup re sexp_pretty ppx_jane mdx

# install owl-symbolic

RUN opam update -q && opam pin --dev-repo owl-symbolic

# HACK: The repositories for older releases that are not supported 
#  (like 11.04, 11.10 and 13.04) get moved to an archive server. 
#  There are repositories available at http://old-releases.ubuntu.com.

RUN sed -i -re 's/([a-z]{2}\.)?archive.ubuntu.com|security.ubuntu.com/old-releases.ubuntu.com/g' /etc/apt/sources.list
RUN apt-get update && apt-get upgrade

# install ode related stuff

RUN apt-get install -y gfortran libsundials-dev
RUN opam pin --dev-repo owl-ode --ignore-constraints-on owl,owl-base

# install owl-tutorials

WORKDIR /home/opam/book
COPY . ${WORKDIR}

ENTRYPOINT /bin/bash
