// Pure-function tests for the prompt-template helpers in admin.html.
// Run: node tests/prompt_template_test.mjs
// The helpers are embedded in admin.html inside a <script>; this test extracts
// them by marker comments so the tested code IS the shipped code.

import { readFileSync } from "node:fs";

const html = readFileSync(new URL("../gateway/static/admin.html", import.meta.url), "utf8");
const m = html.match(/\/\/ PROMPTS-START([\s\S]*?)\/\/ PROMPTS-END/);
if (!m) { console.error("FAIL: PROMPTS markers not found"); process.exit(1); }
const { modelPromptCellText, promptButtonLabel, promptFieldState } = new Function(
  m[1] + "\nreturn { modelPromptCellText, promptButtonLabel, promptFieldState };")();

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

// —— 模型编辑弹窗里「系统提示词」字段的状态 ——
// injected:   已注入模板 → 输入框只读，模板名以标签形式展示
// buttonLabel: 注入/取消注入按钮文案
// buttonDisabled: 新增模型时不可点（模型还没写进 config，注入必然失败）
eq("edit + injected",
   promptFieldState({ isEdit: true, model: { prompt_template: "tpl", system_prompt: "hand" } }),
   { injected: true, readOnly: true, buttonLabel: "取消注入", buttonDisabled: false,
     tag: "tpl", text: "" });
eq("edit + not injected",
   promptFieldState({ isEdit: true, model: { prompt_template: null, system_prompt: "hand" } }),
   { injected: false, readOnly: false, buttonLabel: "注入提示词", buttonDisabled: false,
     tag: null, text: "hand" });
eq("edit + not injected + empty prompt",
   promptFieldState({ isEdit: true, model: { prompt_template: null, system_prompt: null } }),
   { injected: false, readOnly: false, buttonLabel: "注入提示词", buttonDisabled: false,
     tag: null, text: "" });
eq("create mode disables injection",
   promptFieldState({ isEdit: false, model: null }),
   { injected: false, readOnly: false, buttonLabel: "注入提示词", buttonDisabled: true,
     tag: null, text: "" });
eq("create mode ignores a passed model",
   promptFieldState({ isEdit: false, model: { prompt_template: "tpl" } }).injected,
   false);

process.exit(failed ? 1 : 0);
