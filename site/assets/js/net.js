/* ============================================================
 * Neural network visualization (homepage + network page)
 * ------------------------------------------------------------
 * Canvas 2D rendering of the policy network
 * (8 → 64 → 64 → 2, drawn schematically).
 * Pulses travel input → output, echoing a forward pass.
 * Pauses offscreen / hidden tab / reduced motion.
 * ============================================================ */

(function () {
  "use strict";

  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

  const figures = document.querySelectorAll("[data-nn-viz]");
  if (!figures.length) return;

  const css = getComputedStyle(document.documentElement);
  const COLOR_INK = (css.getPropertyValue("--ink-2") || "#4c4636").trim();
  const COLOR_LINE = (css.getPropertyValue("--line-strong") || "#b9af95").trim();
  const COLOR_ACCENT = (css.getPropertyValue("--accent") || "#b45309").trim();
  const COLOR_MONO = (css.getPropertyValue("--ink-3") || "#7c7461").trim();

  /* Real architecture, drawn with representative node counts */
  const LAYERS = [
    { units: 8, drawn: 8, label: "8", sub: "state" },
    { units: 64, drawn: 4, label: "64", sub: "relu" },
    { units: 64, drawn: 4, label: "64", sub: "relu" },
    { units: 2, drawn: 2, label: "2", sub: "Q-values" },
  ];

  figures.forEach((figure) => {
    const canvas = figure.querySelector("canvas");
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    const W = 720;
    const H = 360;
    const MARGIN_X = 70;
    const CENTER_Y = 180;
    const NODE_GAP = 34;

    let dpr = 1;
    let nodes = [];
    let edges = [];
    let pulses = [];
    let running = false;
    let rafId = null;
    let lastSpawn = 0;

    function buildGraph() {
      nodes = LAYERS.map((layer, li) => {
        const x = MARGIN_X + li * ((W - 2 * MARGIN_X) / (LAYERS.length - 1));
        return Array.from({ length: layer.drawn }, (_, ni) => ({
          x,
          y: CENTER_Y + (ni - (layer.drawn - 1) / 2) * NODE_GAP,
          layer: li,
          glow: 0,
        }));
      });

      edges = [];
      for (let li = 1; li < LAYERS.length; li++) {
        nodes[li].forEach((to, ti) => {
          nodes[li - 1].forEach((from, fi) => {
            /* sample edges so dense layers stay readable */
            if ((fi + ti) % 2 === 0 || LAYERS[li - 1].drawn <= 3) {
              edges.push({ from, to, layer: li });
            }
          });
        });
      }
    }

    function resize() {
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = W * dpr;
      canvas.height = H * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function spawnPulse(now) {
      if (now - lastSpawn < 650) return;
      lastSpawn = now;
      /* a forward pass: input node → random path → output */
      const path = [Math.floor(Math.random() * LAYERS[0].drawn)];
      for (let li = 1; li < LAYERS.length; li++) {
        path.push(Math.floor(Math.random() * LAYERS[li].drawn));
      }
      pulses.push({ path, segment: 0, t: 0 });
    }

    function nodeAt(layerIndex, nodeIndex) {
      return nodes[layerIndex][nodeIndex];
    }

    function frame(now) {
      if (!running) return;

      ctx.clearRect(0, 0, W, H);

      /* edges */
      ctx.lineWidth = 0.8;
      ctx.strokeStyle = COLOR_LINE;
      ctx.globalAlpha = 0.55;
      edges.forEach(({ from, to }) => {
        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.lineTo(to.x, to.y);
        ctx.stroke();
      });
      ctx.globalAlpha = 1;

      /* pulses */
      spawnPulse(now);
      pulses = pulses.filter((pulse) => pulse.segment < pulse.path.length - 1);
      pulses.forEach((pulse) => {
        pulse.t += 0.055;
        if (pulse.t >= 1) {
          const arrived = nodeAt(pulse.segment + 1, pulse.path[pulse.segment + 1]);
          if (arrived) arrived.glow = 1;
          pulse.segment += 1;
          pulse.t = 0;
          if (pulse.segment >= pulse.path.length - 1) return;
        }
        const a = nodeAt(pulse.segment, pulse.path[pulse.segment]);
        const b = nodeAt(pulse.segment + 1, pulse.path[pulse.segment + 1]);
        if (!a || !b) return;
        const x = a.x + (b.x - a.x) * pulse.t;
        const y = a.y + (b.y - a.y) * pulse.t;
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fillStyle = COLOR_ACCENT;
        ctx.fill();
      });

      /* nodes */
      nodes.flat().forEach((node) => {
        if (node.glow > 0.01) {
          ctx.beginPath();
          ctx.arc(node.x, node.y, 7 + node.glow * 5, 0, Math.PI * 2);
          ctx.fillStyle = COLOR_ACCENT;
          ctx.globalAlpha = node.glow * 0.25;
          ctx.fill();
          ctx.globalAlpha = 1;
          node.glow *= 0.92;
        }
        ctx.beginPath();
        ctx.arc(node.x, node.y, 6.5, 0, Math.PI * 2);
        ctx.fillStyle = "#f6f2e9";
        ctx.fill();
        ctx.lineWidth = 1.4;
        ctx.strokeStyle = COLOR_INK;
        ctx.stroke();
      });

      /* labels */
      ctx.font = "600 13px 'IBM Plex Mono', monospace";
      ctx.fillStyle = COLOR_INK;
      ctx.textAlign = "center";
      LAYERS.forEach((layer, li) => {
        const x = MARGIN_X + li * ((W - 2 * MARGIN_X) / (LAYERS.length - 1));
        ctx.fillText(layer.label, x, 336);
      });
      ctx.font = "10px 'IBM Plex Mono', monospace";
      ctx.fillStyle = COLOR_MONO;
      LAYERS.forEach((layer, li) => {
        const x = MARGIN_X + li * ((W - 2 * MARGIN_X) / (LAYERS.length - 1));
        ctx.fillText(layer.sub.toUpperCase(), x, 26);
      });

      rafId = requestAnimationFrame(frame);
    }

    function start() {
      if (running) return;
      running = true;
      rafId = requestAnimationFrame(frame);
    }

    function stop() {
      running = false;
      if (rafId) cancelAnimationFrame(rafId);
      rafId = null;
    }

    buildGraph();
    resize();
    window.addEventListener("resize", resize);

    /* draw one static frame so the canvas is never blank */
    running = true;
    frame(0);
    stop();

    if ("IntersectionObserver" in window) {
      const io = new IntersectionObserver(
        (entries) => entries.forEach((e) => (e.isIntersecting ? start() : stop())),
        { threshold: 0.05 }
      );
      io.observe(figure);
    } else {
      start();
    }

    document.addEventListener("visibilitychange", () => {
      if (document.hidden) stop();
    });
  });
})();
