FROM owlbarn/owl:latest
USER opam

WORKDIR /home/opam/owl_tutorial
COPY . ${WORKDIR}
RUN opam install core async lambdasoup re sexp_pretty ppx_jane mdx
RUN sudo make
