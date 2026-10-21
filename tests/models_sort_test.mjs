// Pure-function tests for the models-tab sorting helper in admin.html.
// Run: node tests/models_sort_test.mjs
// The helper is embedded in admin.html inside a <script>; this test extracts
// it by marker comments so the tested code IS the shipped code.

import { readFileSync } from "node:fs";

const html = readFileSync(new URL("../gateway/static/admin.html", import.meta.url), "utf8");
const m = html.match(/\/\/ MODELS-SORT-START([\s\S]*?)\/\/ MODELS-SORT-END/);
if (!m) {
  console.error("FAIL: MODELS-SORT markers not found in admin.html — helper not implemented yet");
  process.exit(1);
}

const { sortModelRows, nextSortState } = new Function(
  m[1] + "\nreturn { sortModelRows, nextSortState };")();

let failed = 0;
function eq(name, actual, expected) {
  const a = JSON.stringify(actual), e = JSON.stringify(expected);
  if (a === e) { console.log("ok  " + name); }
  else { console.log("FAIL " + name + "\n  actual:   " + a + "\n  expected: " + e); failed++; }
}

// rows mirror renderModels' input: {model: {name, protocol, ...}, baseUrl}
function row(name, baseUrl, protocol, upstream_model, system_prompt) {
  return { model: { name, protocol: protocol || null,
                    upstream_model: upstream_model || null, system_prompt: system_prompt || null },
           baseUrl };
}

const ROWS = [
  row("m1", "https://bbb.example", "openai_chat"),
  row("m2", "https://aaa.example", null),
  row("m3", "https://aaa.example", "anthropic_messages"),
  row("m4", "https://ccc.example", "openai_chat"),
];

// --- sort by endpoint (接入点) ---
eq("by endpoint asc groups aaa before bbb before ccc",
   sortModelRows(ROWS, "endpoint", "asc").map(r => r.model.name),
   ["m2", "m3", "m1", "m4"]);
eq("by endpoint desc reverses groups, keeps name order inside",
   sortModelRows(ROWS, "endpoint", "desc").map(r => r.model.name),
   ["m4", "m1", "m2", "m3"]);

// --- sort by protocol (协议) ---
// 缺值（未标记协议，显示 —）视作比任何值都大：升序沉底，降序置顶（纯反转语义）
eq("by protocol asc: missing last, protocols alphabetical",
   sortModelRows(ROWS, "protocol", "asc").map(r => r.model.name),
   ["m3", "m1", "m4", "m2"]);
// desc: missing first, protocols reverse; m1/m4 share openai_chat -> name tiebreak keeps m1 first
eq("by protocol desc: missing first, protocols reverse",
   sortModelRows(ROWS, "protocol", "desc").map(r => r.model.name),
   ["m2", "m1", "m4", "m3"]);

// --- sort by name (模型名称) ---
{
  const rows = [row("c", "https://x"), row("a", "https://x"), row("b", "https://x")];
  eq("by name asc", sortModelRows(rows, "name", "asc").map(r => r.model.name), ["a", "b", "c"]);
  eq("by name desc", sortModelRows(rows, "name", "desc").map(r => r.model.name), ["c", "b", "a"]);
}

// --- sort by upstream_model: missing values (—) follow direction ---
{
  const rows = [
    row("b", "https://x", null, "up-b"),
    row("a", "https://x", null, null),
    row("c", "https://x", null, "up-a"),
  ];
  eq("by upstream_model asc: missing last",
     sortModelRows(rows, "upstream_model", "asc").map(r => r.model.name), ["c", "b", "a"]);
  eq("by upstream_model desc: missing first",
     sortModelRows(rows, "upstream_model", "desc").map(r => r.model.name), ["a", "b", "c"]);
}

// --- sort by system_prompt: an injected template name wins over the
// handwritten prompt, mirroring what the column renders. Both rows carry BOTH
// fields so the two implementations disagree on the outcome: template-first
// ascending -> ["b", "a"], handwritten-first ascending -> ["a", "b"].
const PROMPT_SORT_ROWS = [
  { model: { name: "a", prompt_template: "zzz", system_prompt: "aaa" }, baseUrl: "x" },
  { model: { name: "b", prompt_template: null, system_prompt: "bbb" }, baseUrl: "x" },
];
eq("system_prompt column sorts by template name first (asc)",
   sortModelRows(PROMPT_SORT_ROWS, "system_prompt", "asc").map(r => r.model.name), ["b", "a"]);
eq("system_prompt column sorts by template name first (desc)",
   sortModelRows(PROMPT_SORT_ROWS, "system_prompt", "desc").map(r => r.model.name), ["a", "b"]);
// A row with only a handwritten prompt must still sort on that prompt, and a
// row with both must not silently lose the template.
eq("system_prompt falls back to handwritten when no template",
   sortModelRows([
     { model: { name: "a", prompt_template: null, system_prompt: "zzz" }, baseUrl: "x" },
     { model: { name: "b", prompt_template: null, system_prompt: "aaa" }, baseUrl: "x" },
   ], "system_prompt", "asc").map(r => r.model.name), ["b", "a"]);

// --- secondary sort key: model name, always ascending (统一), regardless of
// primary key/direction and regardless of original order (explicit tiebreak,
// stronger than Array.sort stability) ---
{
  // config order m5,m1 (NOT name order); same endpoint -> tiebreak by name
  const rows = [
    row("m5", "https://x.example", null),
    row("m1", "https://x.example", null),
    row("m0", "https://a.example", null),
  ];
  eq("endpoint tie: name tiebreak overrides config order",
     sortModelRows(rows, "endpoint", "asc").map(r => r.model.name), ["m0", "m1", "m5"]);
  // desc too: group order reverses, names inside the group stay asc
  eq("endpoint desc: names inside the group still asc",
     sortModelRows(rows, "endpoint", "desc").map(r => r.model.name), ["m1", "m5", "m0"]);
}
{
  // protocol tie: two models share openai_chat, config order m4,m1
  const rows = [
    row("m4", "https://x", "openai_chat"),
    row("m1", "https://x", "openai_chat"),
  ];
  eq("protocol tie: name tiebreak overrides config order",
     sortModelRows(rows, "protocol", "asc").map(r => r.model.name), ["m1", "m4"]);
  eq("protocol desc: names inside the group still asc",
     sortModelRows(rows, "protocol", "desc").map(r => r.model.name), ["m1", "m4"]);
}
{
  // both missing primary values -> name tiebreak decides
  const rows = [row("z", "https://x", null), row("a", "https://x", null)];
  eq("both missing: name tiebreak",
     sortModelRows(rows, "protocol", "asc").map(r => r.model.name), ["a", "z"]);
}

// --- header-click state machine: none -> asc -> desc -> none ---
eq("first click sorts asc",
   nextSortState({ key: "default", dir: "asc" }, "endpoint"), { key: "endpoint", dir: "asc" });
eq("second click flips to desc",
   nextSortState({ key: "endpoint", dir: "asc" }, "endpoint"), { key: "endpoint", dir: "desc" });
eq("third click restores default order",
   nextSortState({ key: "endpoint", dir: "desc" }, "endpoint"), { key: "default", dir: "asc" });
eq("clicking another column restarts at asc",
   nextSortState({ key: "endpoint", dir: "desc" }, "protocol"), { key: "protocol", dir: "asc" });

// --- unknown key: copy, never throw, never reorder, never mutate ---
eq("invalid key returns original order (copy)",
   sortModelRows(ROWS, "garbage", "asc").map(r => r.model.name),
   ["m1", "m2", "m3", "m4"]);
eq("invalid key does not mutate input",
   sortModelRows(ROWS, "garbage", "asc") === ROWS, false);
eq("unknown column key returns default state",
   nextSortState({ key: "default", dir: "asc" }, "garbage"), { key: "default", dir: "asc" });

// --- empty input ---
eq("empty rows stay empty", sortModelRows([], "endpoint", "asc"), []);

// --- immutability: input array untouched ---
{
  const input = [row("b", "https://x", null), row("a", "https://y", null)];
  const snapshot = JSON.stringify(input);
  sortModelRows(input, "endpoint", "asc");
  eq("input array not mutated", JSON.stringify(input), snapshot);
}

// renderModels must iterate the SORTED array (regression: the tbody loop
// iterated `rows` while `sorted` was computed and thrown away — clicking
// the header rebuilt the table in config order, so sorting never took effect).
{
  const fn = html.match(/function renderModels\(content\) \{([\s\S]*?)\n\}/);
  if (!fn) {
    console.log("FAIL  renderModels not found in admin.html"); failed++;
  } else {
    const body = fn[1];
    if (!/of sorted\b/.test(body)) {
      console.log("FAIL  renderModels does not iterate the sorted array (no 'of sorted')");
      failed++;
    } else {
      console.log("ok  renderModels iterates the sorted array");
    }
    if (/of rows\b/.test(body)) {
      console.log("FAIL  renderModels still iterates the unsorted rows array");
      failed++;
    } else {
      console.log("ok  renderModels no longer iterates unsorted rows");
    }
    // sorted must be computed BEFORE the loop consumes it
    const sortAt = body.indexOf("sortModelRows(");
    const loopAt = body.indexOf("of sorted");
    if (!(sortAt !== -1 && loopAt !== -1 && sortAt < loopAt)) {
      console.log("FAIL  sorted computed after the loop that consumes it");
      failed++;
    } else {
      console.log("ok  sorted computed before the consuming loop");
    }
  }
}

if (failed) { console.error("\n" + failed + " test(s) failed"); process.exit(1); }
console.log("\nall models-sort tests passed");
