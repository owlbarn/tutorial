#!/usr/bin/env python3
"""Assemble the online tutorial from converted book chapters.

Standard library only: the tutorial repository runs this on every deployment.
The configuration (TOML) chooses which chapters are published and in what
order. References to chapters that are not published become plain text.

    python3 assemble.py --content book --config site.toml --out _site
"""
import argparse
import html
import json
import re
import shutil
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent


def label(number):
    """Chapter numbers are padded (07); appendix letters stay as they are (A)."""
    return number.zfill(2) if number.isdigit() else number


def shell(config, pages, page, body, toc, turn):
    nav = ''.join(
        f'<a class="book-link{" active" if item is page else ""}" href="{item["slug"]}.html"'
        f'{" aria-current=" + chr(34) + "page" + chr(34) if item is page else ""}>'
        f'<span>{label(item["number"])}</span>{html.escape(item["title"])}</a>' for item in pages)
    toc_html = ''.join(
        f'<a class="{"toc-sub" if entry["level"] > 1 else ""}" href="#{entry["id"]}">{html.escape(entry["text"])}</a>'
        for entry in toc)
    title = html.escape(page['title'] + ' · ' + config['title'] if page else config['title'])
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title><link rel="icon" type="image/svg+xml" href="favicon.svg"><link rel="stylesheet" href="reader.css"><link rel="stylesheet" href="syntax.css"><script defer src="reader.js"></script></head>
<body class="{"reading-page" if page else "home-page"}"><a class="skip-link" href="#reading">Skip to text</a>
<aside class="book-nav"><a class="brand" href="index.html"><img class="brand-mark" src="favicon.svg" width="48" height="48" alt=""><span>{html.escape(config["title"])}</span></a>
<p class="subtitle">{html.escape(config["subtitle"])}</p>
<details class="mobile-menu" open><summary>Contents</summary><nav aria-label="Chapters"><div class="nav-label">CHAPTERS</div>{nav}</nav>
<div class="nav-bottom">{''.join(f'<a href="{html.escape(url)}">{html.escape(text)} ↗</a>' for text, url in config.get("links", {}).items())}</div></details></aside>
<div class="reader-layout"><main id="reading"><div class="page-meta"><span>{html.escape(config["title"]).upper()}</span><span>{html.escape(config["edition"]).upper()}</span></div>
<article>{body}</article><nav class="page-turn" aria-label="Pages">{turn}</nav>
<footer class="page-footer">{config["footer"]}</footer></main>
<aside class="page-toc"><div class="nav-label">ON THIS PAGE</div><nav aria-label="On this page">{toc_html}</nav><a class="back-top" href="#reading">Back to top ↑</a></aside></div></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--content', type=Path, required=True, help='directory written by build.py')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    available = {c['slug'] for c in json.loads((args.content / 'manifest.json').read_text())['chapters']}
    unknown = [slug for slug in config['chapters'] if slug not in available]
    if unknown:
        raise SystemExit(f'Chapters not in the content: {", ".join(unknown)}; available: {", ".join(sorted(available))}')
    pages = [json.loads((args.content / f'{slug}.json').read_text()) for slug in config['chapters']]
    published = {page['slug'] for page in pages}
    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    def body(page):
        # A reference into an unpublished chapter keeps its number, without a link.
        return re.sub(r'<a class="xref" data-chapter="([\w-]+)" href="[^"]*">(.*?)</a>',
                      lambda m: m[0] if m[1] in published else m[2], page['body'], flags=re.S)

    for index, page in enumerate(pages):
        turn = []
        previous = pages[index - 1] if index else None
        turn.append(f'<a href="{previous["slug"]}.html"><small>PREVIOUS</small>← {html.escape(previous["title"])}</a>'
                    if previous else '<a href="index.html"><small>OVERVIEW</small>← Contents</a>')
        if index + 1 < len(pages):
            following = pages[index + 1]
            turn.append(f'<a class="next" href="{following["slug"]}.html"><small>NEXT</small>{html.escape(following["title"])} →</a>')
        kind = 'CHAPTER' if page['number'].isdigit() else 'APPENDIX'
        heading = (f'<p class="chapter-number">{kind} {label(page["number"])}</p>'
                   f'<h1>{html.escape(page["title"])}</h1>')
        (args.out / f'{page["slug"]}.html').write_text(
            shell(config, pages, page, heading + body(page), page['toc'], ''.join(turn)))

    contents = ''.join(
        f'<li><a href="{page["slug"]}.html"><span class="contents-number">{label(page["number"])}</span>'
        f'<span>{html.escape(page["title"])}</span><span aria-hidden="true">→</span></a></li>' for page in pages)
    home = (HERE / 'home.html').read_text().replace('{{intro}}', config['intro']).replace(
        '{{title}}', html.escape(config['title'])).replace('{{subtitle}}', html.escape(config['subtitle'])).replace(
        '{{first}}', f'{pages[0]["slug"]}.html' if pages else '#').replace(
        '{{contents}}', f'<section class="contents-group"><ul>{contents}</ul></section>')
    (args.out / 'index.html').write_text(shell(config, pages, None, home, [], ''))

    shutil.copytree(args.content / 'assets', args.out / 'assets')
    shutil.copyfile(args.content / 'syntax.css', args.out / 'syntax.css')
    for name in ('reader.css', 'reader.js', 'favicon.svg'):
        shutil.copyfile(HERE / name, args.out / name)
    # Old URLs keep working: each redirects to its chapter ("slug" or "slug#anchor"),
    # or to the contents when that chapter is not published.
    for old_path, destination in config.get('redirects', {}).items():
        slug, _, fragment = destination.partition('#')
        page_name = f'{slug}.html' + (f'#{fragment}' if fragment else '') if slug in published else 'index.html'
        target = '../' * old_path.count('/') + page_name
        page = args.out / old_path
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(f'<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0; url={target}">'
                        f'<link rel="canonical" href="{target}"><a href="{target}">Moved to {target}</a>\n')
    (args.out / '.nojekyll').touch()
    check(args.out)
    print(f'Assembled {len(pages)} chapters into {args.out}')


def check(out):
    """Every local link and fragment must resolve; ids must be unique per page."""
    pages = {str(p.relative_to(out)): p.read_text() for p in out.rglob('*.html')}
    problems = []
    for name, text in pages.items():
        ids = re.findall(r'\sid=["\']([^"\']+)', text)
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            problems.append(f'{name}: duplicate ids {sorted(duplicates)[:5]}')
        for url in re.findall(r'(?:href|src)="([^"]+)"', text):
            if re.match(r'\w+:|//', url):
                continue
            path, _, fragment = url.partition('#')
            path = str((out / name).parent.joinpath(path).resolve().relative_to(out.resolve())) if path else name
            if not (out / path).exists():
                problems.append(f'{name}: missing {url}')
            elif fragment and path in pages and not re.search(rf'\sid=["\']{re.escape(fragment)}["\']', pages[path]):
                problems.append(f'{name}: missing anchor {url}')
    if problems:
        raise SystemExit(f'{len(problems)} broken links or ids:\n' + '\n'.join(problems))


if __name__ == '__main__':
    main()
