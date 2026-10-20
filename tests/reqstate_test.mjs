// Pure-function tests for the requests-filter state persistence in admin.html.
// Run: node tests/reqstate_test.mjs
// The helpers are embedded in admin.html inside a <script>; this test extracts
// them by marker comments so the tested code IS the shipped code.

import { readFileSync } from "node:fs";

const html = readFileSync(new URL("../gateway/static/admin.html", import.meta.url), "utf8");
const m = html.match(/\/\/ REQSTATE-START([\s\S]*?)\/\/ REQSTATE-END/);
if (!m) {
  console.error("FAIL: REQSTATE markers not found in admin.html — helpers not implemented yet");
  process.exit(1);
}

const { encodeRequestsState, decodeRequestsState } =
  new Function(m[1] + "\nreturn { encodeRequestsState, decodeRequestsState };")();

let failed = 0;
function eq(name, actual, expected) {
  const a = JSON.stringify(actual), e = JSON.stringify(expected);
  if (a === e) { console.log("ok  " + name); }
  else { console.log("FAIL " + name + "\n  actual:   " + a + "\n  expected: " + e); failed++; }
}

// encode: 空值归一为 ""，只保留四个筛选字段
eq("encode full state", encodeRequestsState({
  baseUrl: "https://a.com", model: "m1", from: "2026-08-26T00:00", to: "2026-08-26T23:59",
  limit: 50, offset: 100, total: 9,   // 分页字段不持久化
}), '{"baseUrl":"https://a.com","model":"m1","from":"2026-08-26T00:00","to":"2026-08-26T23:59"}');

eq("encode empty-ish state", encodeRequestsState({ baseUrl: null, model: undefined, from: "", to: "" }),
  '{"baseUrl":"","model":"","from":"","to":""}');

// decode: roundtrip
eq("decode roundtrip", decodeRequestsState(
  encodeRequestsState({ baseUrl: "https://a.com", model: "m1", from: "f", to: "t" })),
  { baseUrl: "https://a.com", model: "m1", from: "f", to: "t" });

// decode: 非法输入一律 null（调用方据此跳过恢复）
eq("decode garbage -> null", decodeRequestsState("not json"), null);
eq("decode null -> null", decodeRequestsState(null), null);
eq("decode array -> null", decodeRequestsState("[1,2]"), null);
eq("decode non-string fields -> empty strings", decodeRequestsState(
  '{"baseUrl":1,"model":true,"from":{},"to":[]}'),
  { baseUrl: "", model: "", from: "", to: "" });

if (failed) { console.error("\n" + failed + " test(s) failed"); process.exit(1); }
console.log("all ok");

// —— 接入点 → 模型列表联动 ——
const { modelsForBaseUrl } = new Function(m[1] + "\nreturn { modelsForBaseUrl };")();

eq("all models when no base_url",
  modelsForBaseUrl("", [{ base_url: "https://a.com", models: [{ name: "m1" }, { name: "m2" }] },
                        { base_url: "https://b.com", models: [{ name: "m3" }] }]),
  ["m1", "m2", "m3"]);

eq("only the endpoint's models when base_url set",
  modelsForBaseUrl("https://b.com", [{ base_url: "https://a.com", models: [{ name: "m1" }] },
                                     { base_url: "https://b.com", models: [{ name: "m3" }, { name: "m4" }] }]),
  ["m3", "m4"]);

eq("unknown base_url -> empty", modelsForBaseUrl("https://nope.com", []), []);
eq("missing endpoints -> empty", modelsForBaseUrl("https://a.com", undefined), []);

// 模型选择有效性：切接入点后，不在新列表中的当前模型要清空
const { keepIfListed } = new Function(m[1] + "\nreturn { keepIfListed };")();
eq("keep model when still listed", keepIfListed("m1", ["m1", "m2"]), "m1");
eq("clear model when not listed", keepIfListed("m1", ["m2"]), "");
eq("empty list clears any model", keepIfListed("m1", []), "");
eq("no current model stays empty", keepIfListed("", ["m1"]), "");
