.PHONY: all clean dep promote test test-all depext push compile cloc

all: compile
	git add docs

compile: test
	-dune build @site
	@echo Site has been generated in _build/default/static/
	cp -r _build/default/static/* docs/

test: tool
	-dune build @runtest
	-dune promote
	@dune exec -- otb-dep $(CURDIR)

test-all:
	@dune build @runtest-all

tool:
	@dune build @install

promote:
	@dune promote

clean:
	@dune clean

push:
	git commit -am "editing book ..." && \
	git push origin `git branch | grep \* | cut -d ' ' -f2`

loc:
	cloc .