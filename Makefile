.PHONY: all clean dep publish promote test test-all docker depext

all:
	@dune build @html
	@echo Site has been generated in _build/default/static/

vendor:
	duniverse init rwo `cat pkgs` --pin mdx,https://github.com/Julow/mdx.git,duniverse_mode

test:
	dune runtest

test-all:
	dune build @runtest-all

dep:
	dune exec -- otb-dep

clean:
	dune clean

.PHONY: push
push:
	git commit -am "editing book ..." && \
	git push origin `git branch | grep \* | cut -d ' ' -f2`
