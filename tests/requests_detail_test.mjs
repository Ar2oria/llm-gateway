// Pure-function tests for request-detail protocol/effort labels in admin.html.
// Run: node tests/requests_detail_test.mjs

import { readFileSync } from "node:fs";

const html = readFileSync(new URL("../gateway/static/admin.html", import.meta.url), "utf8");
const m = html.match(/\/\/ REQUESTS-DETAIL-START([\s\S]*?)\/\/ REQUESTS-DETAIL-END/);
if (!m) {
  console.error("FAIL: REQUESTS-DETAIL markers not found in admin.html — helpers not implemented yet");
  process.exit(1);
}

const { protocolFlowLabel, reasoningEffortLabel } =
  new Function(m[1] + "\nreturn { protocolFlowLabel, reasoningEffortLabel };")();

let failed = 0;
function eq(name, actual, expected) {
  const a = JSON.stringify(actual), e = JSON.stringify(expected);
  if (a === e) { console.log("ok  " + name); }
  else { console.log("FAIL " + name + "\n  actual:   " + a + "\n  expected: " + e); failed++; }
}

eq("converted protocol flow",
  protocolFlowLabel({ client_protocol: "anthropic_messages", upstream_protocol: "openai_chat" }),
  "anthropic_messages → openai_chat");
eq("same protocol still shows flow",
  protocolFlowLabel({ client_protocol: "openai_responses", upstream_protocol: "openai_responses" }),
  "openai_responses → openai_responses");
eq("missing protocols for historical row", protocolFlowLabel({}), "—");
eq("partial protocol metadata", protocolFlowLabel({ upstream_protocol: "openai_chat" }), "openai_chat");

eq("reasoning effort", reasoningEffortLabel({ reasoning_effort: "xhigh" }), "xhigh");
eq("budget-formatted effort", reasoningEffortLabel({ reasoning_effort: "budget:4096" }), "budget:4096");
eq("missing reasoning effort", reasoningEffortLabel({ reasoning_effort: null }), "—");

if (failed) { console.error("\n" + failed + " test(s) failed"); process.exit(1); }
console.log("all ok");
