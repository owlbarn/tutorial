.PHONY: all clean dep publish promote test test-all docker depext push compile docker cloc

all: compile
	git add docs

docker:
	docker build -t owlbarn/book:latest .

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
	docker stop book_builder && docker rm book_builder

push:
	git commit -am "editing book ..." && \
	git push origin `git branch | grep \* | cut -d ' ' -f2`

loc:
	cloc .