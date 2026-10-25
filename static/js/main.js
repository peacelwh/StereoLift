// StereoLift project page: lazy, muted, looping videos; tabs for the simulation groups; BibTeX copy.
(function () {
  // Real-robot clips: selectable 1x / 3x playback, 3x by default. Runs first, before the play-on-scroll observer.
  const DEFAULT_RATE = 3;
  const realVideos = Array.from(document.querySelectorAll("video[data-real]"));
  const allButtons = () => Array.from(document.querySelectorAll("[data-speed-all]"));
  function setRate(v, r) {
    v.dataset.rate = String(r);
    v.defaultPlaybackRate = r; // media loads reset playbackRate to this value
    v.playbackRate = r;
    if (v._speedCtl) v._speedCtl.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(Number(b.dataset.rate) === r)));
  }
  function syncGlobal() {
    const rates = new Set(realVideos.map((v) => Number(v.dataset.rate)));
    allButtons().forEach((b) => b.setAttribute("aria-pressed", String(rates.size === 1 && rates.has(Number(b.dataset.speedAll)))));
  }
  realVideos.forEach((v) => {
    let host = v.parentElement;
    if (!host.classList.contains("panel")) {
      const w = document.createElement("div");
      w.className = "vwrap";
      v.replaceWith(w);
      w.appendChild(v);
      host = w;
    }
    const ctl = document.createElement("div");
    ctl.className = "speed";
    ctl.setAttribute("role", "group");
    ctl.setAttribute("aria-label", "Playback speed");
    [1, 3].forEach((r) => {
      const b = document.createElement("button");
      b.type = "button";
      b.dataset.rate = String(r);
      b.textContent = r + "\u00d7";
      b.addEventListener("click", (e) => {
        e.stopPropagation();
        setRate(v, r);
        syncGlobal();
      });
      ctl.appendChild(b);
    });
    host.appendChild(ctl);
    v._speedCtl = ctl;
    setRate(v, DEFAULT_RATE);
    const keep = () => { v.playbackRate = Number(v.dataset.rate || DEFAULT_RATE); };
    v.addEventListener("loadedmetadata", keep);
    v.addEventListener("play", keep);
  });
  allButtons().forEach((b) =>
    b.addEventListener("click", () => {
      const r = Number(b.dataset.speedAll);
      realVideos.forEach((v) => setRate(v, r));
      syncGlobal();
    })
  );
  syncGlobal();

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
