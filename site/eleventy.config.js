/* ============================================================
 * DQN FLAPPY BIRD — PROJECT WEBSITE
 * Eleventy configuration
 * ------------------------------------------------------------
 * Static site generated for GitHub Pages project hosting:
 *   https://danrosselli.github.io/dqn-flappy-bird/
 * All internal links and assets must respect the path prefix.
 * ============================================================ */

import markdownIt from "markdown-it";
import markdownItAnchor from "markdown-it-anchor";

const PATH_PREFIX = "/dqn-flappy-bird/";

export default function (eleventyConfig) {
  /* ------------------------------------------------------------
   * Passthrough assets (paths relative to this config file)
   * ---------------------------------------------------------- */
  eleventyConfig.addPassthroughCopy({ "assets": "assets" });
  eleventyConfig.addPassthroughCopy({ ".nojekyll": ".nojekyll" });

  /* ------------------------------------------------------------
   * Markdown
   * ---------------------------------------------------------- */
  const md = markdownIt({
    html: true,
    linkify: true,
    typographer: false,
  }).use(markdownItAnchor, {
    permalink: false,
    tabIndex: false,
  });
  eleventyConfig.setLibrary("md", md);

  /* ------------------------------------------------------------
   * Filters
   * ---------------------------------------------------------- */
  eleventyConfig.addFilter("num", (value) => {
    if (value === null || value === undefined) return value;
    const n = Number(value);
    if (Number.isNaN(n)) return value;
    return n.toLocaleString("en-US");
  });

  eleventyConfig.addFilter("fixed", (value, digits = 2) => {
    const n = Number(value);
    if (Number.isNaN(n)) return value;
    return n.toFixed(digits);
  });

  eleventyConfig.addFilter("pad", (value, size = 3) =>
    String(value).padStart(size, "0")
  );

  eleventyConfig.addFilter("dateISO", (value) => {
    if (!value) return "";
    return new Date(value).toISOString().slice(0, 10);
  });

  eleventyConfig.addFilter("json", (value) => JSON.stringify(value));

  /* Find the index of an object in an array by key. */
  eleventyConfig.addFilter("findIndex", (arr, key, value) => {
    if (!Array.isArray(arr)) return -1;
    return arr.findIndex((item) => item[key] === value);
  });

  /* Human-friendly rendering of a config value (objects → JSON). */
  eleventyConfig.addFilter("configVal", (value) => {
    if (value === null || value === undefined) return "—";
    if (typeof value === "object") return JSON.stringify(value);
    return String(value);
  });

  eleventyConfig.addFilter("dateDisplay", (value) => {
    if (!value) return "";
    return new Date(value).toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "2-digit",
      timeZone: "UTC",
    });
  });

  /* Returns the page of a collection currently being rendered,
     plus its previous / next siblings (used by article layout). */
  eleventyConfig.addFilter("withSiblings", (collection, pageUrl) => {
    const index = collection.findIndex((item) => item.url === pageUrl);
    if (index === -1) return { current: null, prev: null, next: null };
    return {
      current: collection[index],
      prev: index > 0 ? collection[index - 1] : null,
      next: index < collection.length - 1 ? collection[index + 1] : null,
    };
  });

  /* ------------------------------------------------------------
   * Collections
   * ---------------------------------------------------------- */
  const byOrder = (a, b) => (a.data.order ?? 99) - (b.data.order ?? 99);

  eleventyConfig.addCollection("howItWorks", (api) =>
    api.getFilteredByTag("how-it-works").sort(byOrder)
  );

  eleventyConfig.addCollection("training", (api) =>
    api.getFilteredByTag("training").sort(byOrder)
  );

  eleventyConfig.addCollection("insights", (api) =>
    api.getFilteredByTag("insight").sort(byOrder)
  );

  /* ------------------------------------------------------------
   * Dev server
   * ---------------------------------------------------------- */
  eleventyConfig.setServerOptions({
    port: 8081,
    showAllHosts: false,
  });

  return {
    pathPrefix: PATH_PREFIX,
    dir: {
      input: "src",
      includes: "_includes",
      data: "_data",
      output: "_site",
    },
    markdownTemplateEngine: "njk",
    htmlTemplateEngine: "njk",
    templateFormats: ["md", "njk", "html"],
  };
}

export const config = {
  pathPrefix: PATH_PREFIX,
};
