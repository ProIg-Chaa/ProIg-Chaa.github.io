const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function setupReadingProgress() {
  const article = document.querySelector(".reading-scope");
  const bar = document.querySelector(".reading-progress span");

  if (!article || !bar) return;

  const update = () => {
    const rect = article.getBoundingClientRect();
    const scrollable = Math.max(1, rect.height - window.innerHeight);
    const read = Math.min(Math.max(-rect.top / scrollable, 0), 1);
    bar.style.transform = `scaleX(${read})`;
  };

  update();
  window.addEventListener("scroll", update, { passive: true });
  window.addEventListener("resize", update);
}

function setupBackToTop() {
  const button = document.querySelector(".back-to-top");
  if (!button) return;

  const update = () => {
    button.classList.toggle("is-visible", window.scrollY > 560);
  };

  button.addEventListener("click", () => {
    window.scrollTo({ top: 0, behavior: reduceMotion ? "auto" : "smooth" });
  });

  update();
  window.addEventListener("scroll", update, { passive: true });
}

function setupTocHighlight() {
  const links = Array.from(document.querySelectorAll("[data-toc-link]"));
  if (!links.length) return;

  const linkById = new Map(
    links.map((link) => [decodeURIComponent(link.hash.slice(1)), link])
  );
  const headings = Array.from(document.querySelectorAll(".article-content :is(h2, h3)"))
    .filter((heading) => linkById.has(heading.id));

  if (!headings.length) return;

  const setActive = (id) => {
    links.forEach((link) => {
      link.classList.toggle("is-active", decodeURIComponent(link.hash.slice(1)) === id);
    });
  };

  const observer = new IntersectionObserver(
    (entries) => {
      const visible = entries
        .filter((entry) => entry.isIntersecting)
        .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);

      if (visible[0]) setActive(visible[0].target.id);
    },
    {
      rootMargin: "-18% 0px -68% 0px",
      threshold: [0, 1]
    }
  );

  headings.forEach((heading) => observer.observe(heading));
  setActive(headings[0].id);
}

function setupRevealMotion() {
  if (reduceMotion) {
    document.documentElement.classList.add("reduce-motion");
    return;
  }

  const items = document.querySelectorAll(
    ".hero-copy, .hero-panel, .section-heading, .note-card, .note-row, .topic-list a, .about-block"
  );

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-revealed");
          observer.unobserve(entry.target);
        }
      });
    },
    { rootMargin: "0px 0px -8% 0px", threshold: 0.12 }
  );

  items.forEach((item) => {
    item.classList.add("reveal");
    observer.observe(item);
  });
}

setupReadingProgress();
setupBackToTop();
setupTocHighlight();
setupRevealMotion();
