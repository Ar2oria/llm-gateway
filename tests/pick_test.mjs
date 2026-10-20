// Pure-function tests for the fetch-models picker helpers in admin.html.
// Run: node tests/pick_test.mjs
// The helpers are embedded in admin.html inside a <script>; this test extracts
// them by marker comments so the tested code IS the shipped code.

import { readFileSync } from "node:fs";

const html = readFileSync(new URL("../gateway/static/admin.html", import.meta.url), "utf8");
const m = html.match(/\/\/ PICK-START([\s\S]*?)\/\/ PICK-END/);
if (!m) {
  console.error("FAIL: PICK markers not found in admin.html — helpers not implemented yet");
  process.exit(1);
}

const { initialPickChecked, pickToolbarClass, pickHeadHtml } = new Function(
  m[1] + "\nreturn { initialPickChecked, pickToolbarClass, pickHeadHtml };"
)();
if (typeof pickHeadHtml !== "function") {
  console.error("FAIL: pickHeadHtml not found in PICK block — header builder not extracted yet");
  process.exit(1);
}

let failed = 0;
function eq(name, actual, expected) {
  const a = JSON.stringify(actual), e = JSON.stringify(expected);
  if (a === e) { console.log("ok  " + name); }
  else { console.log("FAIL " + name + "\n  actual:   " + a + "\n  expected: " + e); failed++; }
}

// 获取模型弹窗默认不预选任何模型（含新模型），由用户手动勾选
eq("new model not pre-checked", initialPickChecked("brand-new", new Set()), false);
eq("existing model not pre-checked", initialPickChecked("gpt-4o", new Set(["gpt-4o"])), false);
eq("no system names -> still not pre-checked", initialPickChecked("m1", new Set(["other"])), false);

// 展示「上新」列时工具栏加 has-news 类（水平缩进与行内复选框对齐）
eq("toolbar class with news column", pickToolbarClass(true), "pick-toolbar has-news");
eq("toolbar class without news column", pickToolbarClass(false), "pick-toolbar");

// 上新列表头不写「上新」二字，仅保留空占位列与行内「新」标记对齐
const newsHead = pickHeadHtml(true);
eq("news header keeps empty placeholder column",
   newsHead.includes('<span class="pick-new-col"></span>'), true);
eq("news header drops the 上新 label", newsHead.includes("上新"), false);
eq("header keeps column labels",
   newsHead.includes("模型名称") && newsHead.includes("上游模型名") && newsHead.includes("协议"), true);
eq("plain header has no news column", pickHeadHtml(false).includes("pick-new-col"), false);

if (failed) { console.error("\n" + failed + " test(s) failed"); process.exit(1); }
console.log("all ok");
