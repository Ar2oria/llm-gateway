// Pure-function tests for the model-news badge helpers in admin.html.
// Run: node tests/models_news_test.mjs
// The helpers are embedded in admin.html inside a <script>; this test extracts
// them by marker comments so the tested code IS the shipped code.

import { readFileSync } from "node:fs";

const html = readFileSync(new URL("../gateway/static/admin.html", import.meta.url), "utf8");
const m = html.match(/\/\/ NEWS-START([\s\S]*?)\/\/ NEWS-END/);
if (!m) {
  console.error("FAIL: NEWS markers not found in admin.html — helpers not implemented yet");
  process.exit(1);
}

const { newsBadgeText, isNewModel, hasNewsModels } = new Function(m[1] + "\nreturn { newsBadgeText, isNewModel, hasNewsModels };")();

let failed = 0;
function eq(name, actual, expected) {
  const a = JSON.stringify(actual), e = JSON.stringify(expected);
  if (a === e) { console.log("ok  " + name); }
  else { console.log("FAIL " + name + "\n  actual:   " + a + "\n  expected: " + e); failed++; }
}

eq("badge text with count", newsBadgeText(["a", "b"]), "模型上新 2");
eq("badge text single", newsBadgeText(["only"]), "模型上新 1");
eq("badge text empty -> null (no badge)", newsBadgeText([]), null);
eq("badge text missing -> null", newsBadgeText(undefined), null);

eq("new model marked", isNewModel("m9", ["m9", "m1"]), true);
eq("known model unmarked", isNewModel("m2", ["m9", "m1"]), false);
eq("no news list -> nothing marked", isNewModel("m9", undefined), false);

// 弹窗「上新」列仅当存在新模型时才渲染
eq("news column shown when new models exist", hasNewsModels(["m9"]), true);
eq("news column hidden when no new models", hasNewsModels([]), false);
eq("news column hidden when list missing", hasNewsModels(undefined), false);

if (failed) { console.error("\n" + failed + " test(s) failed"); process.exit(1); }
console.log("all ok");
