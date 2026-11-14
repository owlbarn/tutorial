# Owl Online Tutorial [![deploy](https://github.com/owlbarn/tutorial/actions/workflows/deploy.yml/badge.svg)](https://github.com/owlbarn/tutorial/actions/workflows/deploy.yml)

The online tutorial is hosted at <https://ocaml.xyz/tutorial/>.

Its chapters come from the book *OCaml Scientific Computing*
([owlbarn/springer](https://github.com/owlbarn/springer)). After the book
builds on its main branch, its CI converts the chapters to HTML and commits
them, with the site tools, to `book/` here. Do not edit `book/` by hand: it is
replaced on every update. Change the text in the book repository instead.

## Choosing the published chapters

Edit `chapters` in [`site.toml`](site.toml) and push. The list is the reading
order; `book/manifest.json` lists the available chapters. References to a
chapter that is not published keep their number but are not links. The same
file holds the home page text, footer, sidebar links and redirects from old
page addresses.

## Preview locally

```bash
python3 book/tools/assemble.py --content book --config site.toml --out _site
python3 -m http.server 8000 --directory _site
```

Python 3.11 or later is needed; nothing else is installed.

## Archive

`archive/` holds the previous hand-written Jekyll tutorial. It is kept for
reference and is not published.
