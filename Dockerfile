FROM owlbarn/owl:latest
USER root

# prepare dependency

RUN apt-get update && apt-get install -y pandoc
RUN cd /home/opam/opam-repository && git pull --quiet origin master
RUN opam install core async lambdasoup re sexp_pretty ppx_jane mdx

# install owl-symbolic

RUN opam update -q && opam pin --dev-repo owl-symbolic

# install owl-tutorials

WORKDIR /home/opam/book
COPY . ${WORKDIR}

ENTRYPOINT /bin/bash