// Pure-function tests for SOCKS5 proxy URL helpers in admin.html.
// Run: node tests/proxy_url_test.mjs

import { readFileSync } from "node:fs";

const html = readFileSync(new URL("../gateway/static/admin.html", import.meta.url), "utf8");
const m = html.match(/\/\/ PROXY-URL-START([\s\S]*?)\/\/ PROXY-URL-END/);
if (!m) {
  console.error("FAIL: PROXY-URL markers not found in admin.html — helpers not implemented yet");
  process.exit(1);
}

const { parseSocks5Proxy, buildSocks5Proxy, socks5ProxySummary } = new Function(
  m[1] + "\nreturn { parseSocks5Proxy, buildSocks5Proxy, socks5ProxySummary };")();

let failed = 0;
function eq(name, actual, expected) {
  const a = JSON.stringify(actual), e = JSON.stringify(expected);
  if (a === e) console.log("ok  " + name);
  else { console.log("FAIL " + name + "\n  actual:   " + a + "\n  expected: " + e); failed++; }
}
function throws(name, fn, message) {
  try { fn(); console.log("FAIL " + name + "\n  expected an exception"); failed++; }
  catch (err) {
    if (err.message === message) console.log("ok  " + name);
    else { console.log("FAIL " + name + "\n  actual error: " + err.message + "\n  expected: " + message); failed++; }
  }
}

eq("parse full proxy into separate fields",
   parseSocks5Proxy("socks5://u%40x:p%3Aq@127.0.0.1:1080"),
   { host: "127.0.0.1", port: "1080", username: "u@x", password: "p:q" });
eq("parse proxy without credentials",
   parseSocks5Proxy("socks5://example.com:1080"),
   { host: "example.com", port: "1080", username: "", password: "" });
eq("parse empty proxy",
   parseSocks5Proxy(null),
   { host: "", port: "", username: "", password: "" });
eq("parse invalid proxy as empty",
   parseSocks5Proxy("http://example.com:1080"),
   { host: "", port: "", username: "", password: "" });
eq("build proxy with credentials and URL-encode them",
   buildSocks5Proxy("127.0.0.1", "1080", "u@x", "p:q"),
   "socks5://u%40x:p%3Aq@127.0.0.1:1080");
eq("build proxy without credentials",
   buildSocks5Proxy("example.com", "1080", "", ""),
   "socks5://example.com:1080");
eq("all blank means proxy disabled", buildSocks5Proxy("", "", "", ""), null);
eq("summary for missing proxy", socks5ProxySummary(parseSocks5Proxy(null)), "未配置");
eq("summary for unauthenticated proxy",
   socks5ProxySummary(parseSocks5Proxy("socks5://example.com:1080")),
   "example.com:1080 · 无认证");
eq("summary for authenticated proxy",
   socks5ProxySummary(parseSocks5Proxy("socks5://user:pass@127.0.0.1:1080")),
   "127.0.0.1:1080 · 已认证");
throws("host is required when proxy is enabled", () => buildSocks5Proxy("", "1080", "", ""),
       "SOCKS5 主机不能为空");
throws("port must be numeric", () => buildSocks5Proxy("example.com", "abc", "", ""),
       "SOCKS5 端口必须是 1-65535 的整数");
throws("password requires username", () => buildSocks5Proxy("example.com", "1080", "", "secret"),
       "填写密码时必须填写账号");

if (failed) { console.error("\n" + failed + " test(s) failed"); process.exit(1); }
console.log("\nall proxy URL tests passed");
