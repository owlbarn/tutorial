FROM owlbarn/owl:latest
USER root

# prepare dependency

RUN apt-get update && apt-get install -y pandoc
RUN opam install core async lambdasoup re sexp_pretty ppx_jane mdx

# install owl-symbolic

ENV OWLSYMPATH /home/opam/owl_symbolic
RUN git clone https://github.com/owlbarn/owl_symbolic.git ${OWLSYMPATH}
RUN cd ${OWLSYMPATH} && opam pin .

# install owl-tutorials

WORKDIR /home/opam/owl_tutorials
COPY . ${WORKDIR}

ENTRYPOINT /bin/bash