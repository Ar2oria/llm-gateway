// Pure-function tests for the copy-curl helper in admin.html.
// Run: node tests/curl_test.mjs
// The helper is embedded in admin.html inside a <script>; this test extracts
// it by marker comments so the tested code IS the shipped code.

import { readFileSync } from "node:fs";

const html = readFileSync(new URL("../gateway/static/admin.html", import.meta.url), "utf8");
const m = html.match(/\/\/ CURL-START([\s\S]*?)\/\/ CURL-END/);
if (!m) {
  console.error("FAIL: CURL markers not found in admin.html — helper not implemented yet");
  process.exit(1);
}

const { buildModelCurl } = new Function(m[1] + "\nreturn { buildModelCurl };")();

// shell line continuation between lines: space + backslash + newline + 2 spaces
const NL = " \\\n  ";

let failed = 0;
function eq(name, actual, expected) {
  const a = JSON.stringify(actual), e = JSON.stringify(expected);
  if (a === e) { console.log("ok  " + name); }
  else { console.log("FAIL " + name + "\n  actual:   " + a + "\n  expected: " + e); failed++; }
}

const BASE = "http://127.0.0.1:8088";
const KEY = "sk-sentinel";

// anthropic_messages -> /v1/messages + anthropic-version + max_tokens body
eq("anthropic_messages curl",
   buildModelCurl("anthropic_messages", "claude-sonnet-5", BASE, KEY),
   "curl --compressed -X POST 'http://127.0.0.1:8088/v1/messages'" + NL
   + "-H 'Authorization: Bearer sk-sentinel'" + NL
   + "-H 'anthropic-version: 2023-06-01'" + NL
   + "-H 'Content-Type: application/json'" + NL
   + "-d '{\"model\":\"claude-sonnet-5\",\"max_tokens\":1024,"
   + "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'");

// openai_chat -> /v1/chat/completions + messages body, no anthropic-version
eq("openai_chat curl",
   buildModelCurl("openai_chat", "gpt-4o", BASE, KEY),
   "curl --compressed -X POST 'http://127.0.0.1:8088/v1/chat/completions'" + NL
   + "-H 'Authorization: Bearer sk-sentinel'" + NL
   + "-H 'Content-Type: application/json'" + NL
   + "-d '{\"model\":\"gpt-4o\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'");

// openai_responses -> /v1/responses + input body
eq("openai_responses curl",
   buildModelCurl("openai_responses", "o1", BASE, KEY),
   "curl --compressed -X POST 'http://127.0.0.1:8088/v1/responses'" + NL
   + "-H 'Authorization: Bearer sk-sentinel'" + NL
   + "-H 'Content-Type: application/json'" + NL
   + "-d '{\"model\":\"o1\",\"input\":\"hi\"}'");

// no api key -> no Authorization header at all
eq("empty api key omits Authorization",
   buildModelCurl("openai_chat", "gpt-4o", BASE, ""),
   "curl --compressed -X POST 'http://127.0.0.1:8088/v1/chat/completions'" + NL
   + "-H 'Content-Type: application/json'" + NL
   + "-d '{\"model\":\"gpt-4o\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'");

// unknown protocol -> null (caller shows the picker instead)
eq("unknown protocol returns null", buildModelCurl("nope", "m", BASE, KEY), null);

// model name containing a single quote must be shell-escaped inside -d '...'
// (JSON.stringify does NOT escape single quotes)
eq("single quote in model name is shell-escaped",
   buildModelCurl("openai_responses", "it's", BASE, KEY),
   "curl --compressed -X POST 'http://127.0.0.1:8088/v1/responses'" + NL
   + "-H 'Authorization: Bearer sk-sentinel'" + NL
   + "-H 'Content-Type: application/json'" + NL
   + "-d '{\"model\":\"it'\\''s\",\"input\":\"hi\"}'");

// trailing slash on the base URL is stripped before appending the path
eq("base URL trailing slash stripped",
   buildModelCurl("openai_chat", "m", BASE + "/", KEY).includes("'" + BASE + "/v1/chat/completions'"),
   true);

if (failed) { console.error("\n" + failed + " test(s) failed"); process.exit(1); }
console.log("\nall curl tests passed");
