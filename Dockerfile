FROM owlbarn/owl:latest
USER root

WORKDIR /home/opam/owl_tutorials
COPY . ${WORKDIR}
RUN apt-get update && apt-get install -y pandoc
RUN opam install core async lambdasoup re sexp_pretty ppx_jane mdx

ENTRYPOINT /bin/bash