.PHONY: all clean dep publish promote test test-all docker depext push

all: test
	dune build @html
	echo Site has been generated in _build/default/static/

vendor:
	duniverse init rwo `cat pkgs` --pin mdx,https://github.com/Julow/mdx.git,duniverse_mode

test: tool
	dune build @runtest
	dune exec -- otb-dep $(CURDIR)

test-all:
	dune build @runtest-all

tool:
	dune build @install

clean:
	dune clean

push:
	git commit -am "editing book ..." && \
	git push origin `git branch | grep \* | cut -d ' ' -f2`
