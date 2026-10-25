// StereoLift project page: lazy, muted, looping videos; tabs for the simulation groups; BibTeX copy.
(function () {
  // Play videos only while they are on screen (saves bandwidth with ~30 clips on the page).
  const videos = Array.from(document.querySelectorAll("video[data-autoplay]"));
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        const v = e.target;
        if (e.isIntersecting) {
          if (v.preload === "none") v.preload = "auto";
          v.play().catch(() => {});
        } else {
          v.pause();
        }
      });
    },
    { rootMargin: "120px 0px", threshold: 0.15 }
  );
  videos.forEach((v) => {
    v.muted = true;
    v.loop = true;
    v.playsInline = true;
    io.observe(v);
    // click to toggle play / pause (and to unmute nothing: the clips have no audio)
    v.addEventListener("click", () => (v.paused ? v.play() : v.pause()));
  });

  // Tabs (simulation groups)
  document.querySelectorAll("[data-tabs]").forEach((root) => {
    const tabs = Array.from(root.querySelectorAll(".tab"));
    const panels = tabs.map((t) => document.getElementById(t.getAttribute("aria-controls")));
    function select(i) {
      tabs.forEach((t, k) => t.setAttribute("aria-selected", k === i ? "true" : "false"));
      panels.forEach((p, k) => {
        p.hidden = k !== i;
        if (k === i) p.querySelectorAll("video").forEach((v) => v.play().catch(() => {}));
        else p.querySelectorAll("video").forEach((v) => v.pause());
      });
    }
    tabs.forEach((t, i) => t.addEventListener("click", () => select(i)));
    select(0);
  });

  // BibTeX copy button
  const copyBtn = document.getElementById("copy-bib");
  if (copyBtn) {
    copyBtn.addEventListener("click", async () => {
      const text = document.getElementById("bibtex").innerText;
      try {
        await navigator.clipboard.writeText(text);
        copyBtn.textContent = "Copied";
        setTimeout(() => (copyBtn.textContent = "Copy"), 1500);
      } catch (_) {
        copyBtn.textContent = "Select and copy";
      }
    });
  }
})();
