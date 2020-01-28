.PHONY: all clean dep publish promote test test-all docker depext push compile docker cloc

all:
	-docker run -t -d --name book_builder owlbarn/book:latest
	docker cp . book_builder:/home/opam/book_local
	docker exec -it book_builder bash -c 'cd /home/opam/book_local && export PATH=/home/opam/.cabal/bin:${PATH} && eval `opam env` && make compile'
	docker cp book_builder:/home/opam/book_local/book .
	docker cp book_builder:/home/opam/book_local/docs .
	docker cp book_builder:/home/opam/book_local/static .
	git add docs

docker:
	docker build -t owlbarn/book:latest .

compile: test
	-dune build @site @pdf
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