FROM owlbarn/book:base
USER root

# prepare dependency

RUN cd /home/opam/opam-repository && git pull --quiet origin master
RUN opam install core async lambdasoup re sexp_pretty ppx_jane mdx

# install owl-symbolic

RUN opam update -q && opam pin --dev-repo owl-symbolic

# install ode related stuff

RUN apt-get install -y gfortran libsundials-dev
RUN opam pin https://github.com/owlbarn/owl_ode.git

# install owl-tutorials

WORKDIR /home/opam/book
COPY . ${WORKDIR}

ENTRYPOINT /bin/bash