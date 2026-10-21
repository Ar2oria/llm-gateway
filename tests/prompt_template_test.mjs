// Pure-function tests for the prompt-template helpers in admin.html.
// Run: node tests/prompt_template_test.mjs
// The helpers are embedded in admin.html inside a <script>; this test extracts
// them by marker comments so the tested code IS the shipped code.

import { readFileSync } from "node:fs";

const html = readFileSync(new URL("../gateway/static/admin.html", import.meta.url), "utf8");
const m = html.match(/\/\/ PROMPTS-START([\s\S]*?)\/\/ PROMPTS-END/);
if (!m) { console.error("FAIL: PROMPTS markers not found"); process.exit(1); }
const { modelPromptCellText, promptButtonLabel } = new Function(
  m[1] + "\nreturn { modelPromptCellText, promptButtonLabel };")();

let failed = 0;
function eq(name, actual, expected) {
  const a = JSON.stringify(actual), e = JSON.stringify(expected);
  if (a === e) console.log("ok  " + name);
  else { console.log("FAIL " + name + "\n  actual:   " + a + "\n  expected: " + e); failed++; }
}

eq("template wins over handwritten", modelPromptCellText({ prompt_template: "tpl", system_prompt: "hand" }), "tpl");
eq("handwritten when no template", modelPromptCellText({ prompt_template: null, system_prompt: "hand" }), "hand");
eq("dash when empty", modelPromptCellText({ prompt_template: null, system_prompt: null }), "—");
eq("button label injected", promptButtonLabel({ prompt_template: "tpl" }), "取消注入");
eq("button label not injected", promptButtonLabel({ prompt_template: null }), "注入提示词");

process.exit(failed ? 1 : 0);
