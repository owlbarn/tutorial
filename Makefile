.PHONY: all clean dep publish promote test test-all docker depext push

all: test
	@dune build @site
	@echo Site has been generated in _build/default/static/

test: tool
	@dune build @runtest
	@dune exec -- otb-dep $(CURDIR)

test-all:
	@dune build @runtest-all

tool:
	@dune build @install

clean:
	@dune clean

push:
	git commit -am "editing book ..." && \
	@xgit push origin `git branch | grep \* | cut -d ' ' -f2`
