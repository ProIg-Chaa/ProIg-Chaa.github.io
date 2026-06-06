const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function setupMeteors() {
  const layer = document.querySelector(".meteor-layer");
  if (!layer || reduceMotion) return;

  const meteorCount = window.innerWidth < 720 ? 14 : 28;
  layer.replaceChildren();

  for (let index = 0; index < meteorCount; index += 1) {
    const meteor = document.createElement("span");
    const size = 96 + Math.random() * 168;
    const top = -16 + Math.random() * 76;
    const left = 78 + Math.random() * 34;
    const duration = 3.8 + Math.random() * 3.4;
    const delay = -(index * 0.34 + Math.random() * 0.9);
    const opacity = 0.58 + Math.random() * 0.36;

    meteor.className = "meteor";
    meteor.style.setProperty("--meteor-width", `${size}px`);
    meteor.style.setProperty("--meteor-top", `${top}%`);
    meteor.style.setProperty("--meteor-left", `${left}%`);
    meteor.style.setProperty("--meteor-delay", `${delay}s`);
    meteor.style.setProperty("--meteor-duration", `${duration}s`);
    meteor.style.setProperty("--meteor-opacity", `${opacity}`);
    layer.append(meteor);
  }
}

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
setupMeteors();
