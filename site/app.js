const FILES = [
  "__init__.py",
  "types.py",
  "assemble.py",
  "repair.py",
  "invariants.py",
  "shrink.py",
  "corpus.py",
  "webapp.py",
];

const statusEl = document.getElementById("status");
const go = document.getElementById("go");
const errorEl = document.getElementById("error");
const resultsEl = document.getElementById("results");
const exampleEl = document.getElementById("example");
const formatEl = document.getElementById("format");
const sampleEl = document.getElementById("sample");

let pyodide = null;
let presets = [];

function setStatus(text) {
  statusEl.textContent = text;
}

function showError(text) {
  errorEl.hidden = false;
  errorEl.textContent = text;
}

function clearError() {
  errorEl.hidden = true;
  errorEl.textContent = "";
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function splitLabel(split) {
  if (split === "chunks:1") return "one character at a time";
  const chunks = /^chunks:(\d+)$/.exec(split);
  if (chunks) return `pieces of ${chunks[1]} characters`;
  const cut = /^cut@(\d+)$/.exec(split);
  if (cut) return `cut after ${cut[1]} characters`;
  const trunc = /^trunc@(\d+)$/.exec(split);
  if (trunc) return `only the first ${trunc[1]} characters`;
  return split;
}

function renderRepair(repair) {
  const section = el("section");
  section.append(el("h2", null, "Repaired output"));
  section.append(el("p", null, repair.finish_label || "Done"));
  if (repair.error) {
    section.append(el("p", null, repair.error));
    return section;
  }
  if (repair.content) {
    section.append(el("p", "rowline", "Reply text"));
    section.append(el("pre", null, repair.content));
  }
  if (repair.reasoning) {
    section.append(el("p", "rowline", "Reasoning"));
    section.append(el("pre", null, repair.reasoning));
  }
  if (!repair.calls.length) {
    section.append(el("p", null, "No tool call in this text."));
    return section;
  }
  for (const call of repair.calls) {
    const card = el("article", "call");
    card.append(el("h3", "name", call.name || "(no name)"));
    const bits = [`#${call.index}`, call.id || ""].filter(Boolean).join(" · ");
    card.append(el("p", "meta", bits));
    card.append(el("pre", null, call.arguments_pretty || call.arguments || ""));
    section.append(card);
  }
  return section;
}

function renderFuzz(report) {
  const section = el("section");
  section.append(el("h2", null, "Splitting the repaired read"));
  const fuzz = report.repair_fuzz;
  let sentence = `Checked ${fuzz.splits} ways of cutting this text. `;
  if (fuzz.fail === 0) {
    sentence += "The repaired read stayed the same every time.";
  } else {
    sentence += `${fuzz.fail} checks failed. The repaired reader disagreed with itself.`;
  }
  section.append(el("p", null, sentence));
  if (report.truncated_for_fuzz) {
    section.append(
      el(
        "p",
        null,
        `The split check used the first ${report.fuzz_characters} characters. The repaired output above is the whole paste.`,
      ),
    );
  }
  return section;
}

function renderReader(report) {
  const section = el("section");
  section.append(el("h2", null, "When each piece is read on its own"));
  section.append(
    el(
      "p",
      null,
      "This reader parses each piece separately. A marker cut in half is invisible to it. The repaired reader above holds that marker and finishes the call.",
    ),
  );
  const fuzz = report.split_reader;
  const banner = el("p", "banner");
  if (fuzz.fail === 0) {
    banner.textContent = `Checked ${fuzz.splits} splits. This text reads the same in one piece and in separate pieces.`;
  } else {
    banner.textContent = `Checked ${fuzz.splits} splits. ${fuzz.fail} checks failed when the text was cut into pieces.`;
  }
  section.append(banner);
  if (!fuzz.repros.length) {
    section.append(el("p", null, "No minimal example was stored for this text."));
    return section;
  }
  section.append(el("h2", null, "Smallest examples"));
  for (const repro of fuzz.repros) {
    const card = el("article");
    card.append(el("h3", null, repro.label));
    card.append(el("p", null, repro.explain || repro.detail));
    card.append(el("p", "meta", splitLabel(repro.split)));
    card.append(el("pre", null, repro.shrunk));
    card.append(el("p", "rowline", `One piece — ${repro.one_piece}`));
    card.append(el("p", "rowline", `Split — ${repro.split_up}`));
    section.append(card);
  }
  section.append(
    el(
      "p",
      null,
      "The published vLLM and SGLang results are on the Results matrix tab. This page does not run those parsers.",
    ),
  );
  return section;
}

function render(report) {
  resultsEl.replaceChildren();
  if (report.error) {
    showError(report.error);
    resultsEl.hidden = true;
    return;
  }
  clearError();
  resultsEl.hidden = false;
  resultsEl.append(renderRepair(report.repair), renderFuzz(report), renderReader(report));
}

async function boot() {
  try {
    setStatus("Loading the checker…");
    const py = await import("https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.mjs");
    pyodide = await py.loadPyodide({
      indexURL: "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/",
    });
    pyodide.FS.mkdirTree("/lib/splitcall");
    for (const name of FILES) {
      const response = await fetch(`py/splitcall/${name}`);
      if (!response.ok) throw new Error(`Missing ${name}`);
      pyodide.FS.writeFile(`/lib/splitcall/${name}`, await response.text());
    }
    pyodide.runPython("import sys\nsys.path.insert(0, '/lib')\nfrom splitcall.webapp import analyze, presets_json");
    presets = JSON.parse(String(pyodide.runPython("presets_json()")));
    for (const row of presets) {
      const option = document.createElement("option");
      option.value = row.id;
      option.textContent = row.label;
      exampleEl.append(option);
    }
    go.disabled = false;
    setStatus("Ready. Paste an output, or pick an example.");
  } catch (err) {
    go.disabled = true;
    setStatus("The checker did not load.");
    showError(err && err.message ? err.message : String(err));
  }
}

exampleEl.addEventListener("change", () => {
  const row = presets.find((item) => item.id === exampleEl.value);
  if (!row) return;
  formatEl.value = row.format;
  sampleEl.value = row.text;
});

document.getElementById("checker").addEventListener("submit", (event) => {
  event.preventDefault();
  if (!pyodide) return;
  clearError();
  go.disabled = true;
  setStatus("Checking every way to split this text…");
  resultsEl.hidden = true;
  window.setTimeout(() => {
    try {
      pyodide.globals.set("INPUT_FMT", formatEl.value);
      pyodide.globals.set("INPUT_TEXT", sampleEl.value);
      const report = JSON.parse(String(pyodide.runPython("analyze(INPUT_FMT, INPUT_TEXT)")));
      render(report);
      setStatus("Done.");
    } catch (err) {
      showError(err && err.message ? err.message : String(err));
      setStatus("The check failed.");
    } finally {
      go.disabled = false;
    }
  }, 30);
});

document.getElementById("tab-check").addEventListener("click", () => selectTab("check"));
document.getElementById("tab-matrix").addEventListener("click", () => selectTab("matrix"));

function selectTab(name) {
  const check = name === "check";
  document.getElementById("tab-check").setAttribute("aria-selected", check ? "true" : "false");
  document.getElementById("tab-matrix").setAttribute("aria-selected", check ? "false" : "true");
  document.getElementById("panel-check").hidden = !check;
  document.getElementById("panel-matrix").hidden = check;
  const frame = document.getElementById("matrix-frame");
  if (!check && !frame.getAttribute("src")) frame.src = "matrix.html";
}

boot();
