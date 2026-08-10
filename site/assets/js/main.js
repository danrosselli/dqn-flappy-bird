/* ============================================================
 * DQN FLAPPY BIRD — SITE
 * Progressive enhancement only: the site works without JS.
 * ------------------------------------------------------------
 * - reduced-motion flag
 * - mobile navigation
 * - scroll reveal (IntersectionObserver)
 * - footer year
 * - live-demo availability check
 * ============================================================ */

(function () {
  "use strict";

  const doc = document.documentElement;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

  function applyMotionPref() {
    doc.classList.toggle("reduced-motion", reducedMotion.matches);
  }
  applyMotionPref();
  reducedMotion.addEventListener("change", applyMotionPref);

  /* ------------------------------------------------------------
   * Mobile navigation
   * ---------------------------------------------------------- */
  const toggle = document.querySelector(".nav-toggle");
  const navList = document.getElementById("nav-list");

  if (toggle && navList) {
    const setOpen = (open) => {
      toggle.setAttribute("aria-expanded", String(open));
      navList.classList.toggle("is-open", open);
    };

    toggle.addEventListener("click", () =>
      setOpen(toggle.getAttribute("aria-expanded") !== "true")
    );

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") setOpen(false);
    });

    navList.addEventListener("click", (event) => {
      if (event.target.closest("a")) setOpen(false);
    });

    document.addEventListener("click", (event) => {
      if (!event.target.closest(".main-nav")) setOpen(false);
    });
  }

  /* ------------------------------------------------------------
   * Scroll reveal
   * ---------------------------------------------------------- */
  const revealEls = Array.from(document.querySelectorAll("[data-reveal]"));

  if (revealEls.length && "IntersectionObserver" in window && !reducedMotion.matches) {
    /* stagger siblings that reveal together */
    const groups = new Map();
    revealEls.forEach((el) => {
      const parent = el.parentElement;
      if (!groups.has(parent)) groups.set(parent, 0);
      const index = groups.get(parent);
      groups.set(parent, index + 1);
      el.style.setProperty("--reveal-delay", `${Math.min(index, 5) * 70}ms`);
    });

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.12, rootMargin: "0px 0px -6% 0px" }
    );

    revealEls.forEach((el) => observer.observe(el));
  } else {
    revealEls.forEach((el) => el.classList.add("is-visible"));
  }

  /* ------------------------------------------------------------
   * Footer year
   * ---------------------------------------------------------- */
  document.querySelectorAll("[data-year]").forEach((el) => {
    el.textContent = String(new Date().getFullYear());
  });

  /* ------------------------------------------------------------
   * Live demo — embed the game bundle when CI has produced it
   * ---------------------------------------------------------- */
  const demoFrame = document.querySelector("[data-demo-frame]");

  if (demoFrame) {
    const src = demoFrame.getAttribute("data-demo-src");
    const stage = demoFrame.querySelector(".demo-stage");
    const loading = demoFrame.querySelector("[data-demo-loading]");
    const fallback = demoFrame.querySelector("[data-demo-fallback]");

    const showFallback = () => {
      if (loading) loading.hidden = true;
      if (fallback) fallback.hidden = false;
    };

    const showGame = () => {
      if (!stage) return;
      const iframe = document.createElement("iframe");
      iframe.src = src;
      iframe.title = "DQN Flappy Bird — the agent training live in your browser";
      iframe.loading = "lazy";
      iframe.setAttribute("allow", "autoplay; fullscreen; cross-origin-isolated");
      stage.appendChild(iframe);
      if (loading) loading.hidden = true;
    };

    fetch(src, { method: "HEAD" })
      .then((res) => (res.ok ? showGame() : showFallback()))
      .catch(showFallback);
  }
})();
