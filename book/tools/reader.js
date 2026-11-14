// The pages, figures and mathematics also work with JavaScript disabled.
const menu = document.querySelector('.mobile-menu');
if (window.matchMedia('(max-width: 780px)').matches) menu.open = false;
else {
  const rail = document.querySelector('.book-nav');
  const active = rail.querySelector('[aria-current="page"]');
  if (active && active.offsetTop > rail.clientHeight - 100) {
    rail.scrollTop = active.offsetTop - rail.clientHeight / 2;
  }
}

for (const button of document.querySelectorAll('.copy-code')) {
  button.addEventListener('click', async () => {
    const code = button.closest('.code-card').querySelector('pre code').cloneNode(true);
    code.querySelectorAll('.line-no').forEach(node => node.remove());
    try {
      await navigator.clipboard.writeText(code.textContent);
      button.textContent = 'Copied';
    } catch {
      // Local file previews may not grant clipboard permission.
      button.textContent = 'Select to copy';
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(button.closest('.code-card').querySelector('pre code'));
      selection.removeAllRanges();
      selection.addRange(range);
    }
    setTimeout(() => { button.textContent = 'Copy'; }, 1800);
  });
}
const links = [...document.querySelectorAll('.page-toc nav a')];
if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver(entries => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      links.forEach(link => link.classList.toggle('current', link.hash === `#${entry.target.id}`));
    }
  }, {rootMargin: '-10% 0px -65% 0px'});
  document.querySelectorAll('article h3, article h4').forEach(heading => observer.observe(heading));
}
