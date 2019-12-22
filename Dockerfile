FROM owlbarn/owl:latest
USER root

WORKDIR /home/opam/owl_tutorial
COPY . ${WORKDIR}
RUN apt-get update && apt-get install pandoc
RUN opam install core async lambdasoup re sexp_pretty ppx_jane mdx
#RUN sudo make
