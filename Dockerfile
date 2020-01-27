FROM owlbarn/owl:latest
USER root

# prepare dependency

RUN apt-get update && apt-get install -y cabal-install
RUN cabal update && cabal install pandoc pandoc-citeproc pandoc-crossref
RUN echo 'export PATH=/home/opam/.cabal/bin:${PATH}' >> /home/opam/.bashrc

RUN cd /home/opam/opam-repository && git pull --quiet origin master
RUN opam install core async lambdasoup re sexp_pretty ppx_jane mdx

# install owl-symbolic

RUN opam update -q && opam pin --dev-repo owl-symbolic

# install full latex package

RUN apt-get install -y texlive-full

# install owl-tutorials

WORKDIR /home/opam/book
COPY . ${WORKDIR}

ENTRYPOINT /bin/bash