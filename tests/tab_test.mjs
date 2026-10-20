// Pure-function tests for the tab-persistence helper in admin.html.
// Run: node tests/tab_test.mjs
// The helper is embedded in admin.html inside a <script>; this test extracts
// it by marker comments so the tested code IS the shipped code.

import { readFileSync } from "node:fs";

const html = readFileSync(new URL("../gateway/static/admin.html", import.meta.url), "utf8");
const m = html.match(/\/\/ TABHASH-START([\s\S]*?)\/\/ TABHASH-END/);
if (!m) {
  console.error("FAIL: TABHASH markers not found in admin.html — helper not implemented yet");
  process.exit(1);
}

const { tabFromHash } = new Function(m[1] + "\nreturn { tabFromHash };")();

let failed = 0;
function eq(name, actual, expected) {
  const a = JSON.stringify(actual), e = JSON.stringify(expected);
  if (a === e) { console.log("ok  " + name); }
  else { console.log("FAIL " + name + "\n  actual:   " + a + "\n  expected: " + e); failed++; }
}

// known tabs survive a refresh
eq("#endpoints", tabFromHash("#endpoints"), "endpoints");
eq("#models", tabFromHash("#models"), "models");
eq("#stats", tabFromHash("#stats"), "stats");
eq("#requests (drill-in view)", tabFromHash("#requests"), "requests");

// absent / unknown / garbage -> default tab, never a crash
eq("no hash -> endpoints", tabFromHash(""), "endpoints");
eq("bare # -> endpoints", tabFromHash("#"), "endpoints");
eq("unknown tab -> endpoints", tabFromHash("#garbage"), "endpoints");
eq("prototype-pollution-ish value -> endpoints", tabFromHash("#__proto__"), "endpoints");
eq("unencoded value -> endpoints", tabFromHash("#models%20"), "endpoints");

if (failed) { console.error("\n" + failed + " test(s) failed"); process.exit(1); }
console.log("\nall tab tests passed");
