// Pure-function tests for the client-side minute-bucketing helper in admin.html.
// Run: node tests/bucket_test.mjs
// The helper is embedded in admin.html inside a <script>; this test extracts
// it by marker comments so the tested code IS the shipped code.

import { readFileSync } from "node:fs";

const html = readFileSync(new URL("../gateway/static/admin.html", import.meta.url), "utf8");
const m = html.match(/\/\/ BUCKETING-START([\s\S]*?)\/\/ BUCKETING-END/);
if (!m) {
  console.error("FAIL: BUCKETING markers not found in admin.html — helper not implemented yet");
  process.exit(1);
}
new Function(m[1] + "\nreturn bucketMinutes;")();

let failed = 0;
function eq(name, actual, expected) {
  const a = JSON.stringify(actual), e = JSON.stringify(expected);
  if (a === e) { console.log("ok  " + name); }
  else { console.log("FAIL " + name + "\n  actual:   " + a + "\n  expected: " + e); failed++; }
}

const { bucketMinutes } = new Function(m[1] + "\nreturn { bucketMinutes };")();

const MIN = 60;

// --- bucket width selection ---
// 10 active minutes -> stays 1-minute (well under 120 bars)
{
  const minutes = Array.from({ length: 10 }, (_, i) => ({
    ts: 1786320000 + i * MIN, tokens: { m1: { input: 1 } }, costs: {},
  }));
  eq("few minutes keep 1-min buckets",
     bucketMinutes(minutes).map(b => b.ts),
     minutes.map(x => x.ts));
}

// 331 active minutes -> widens (5-min buckets, 67 bars)
{
  const minutes = [];
  for (let i = 0; i < 331; i++) {
    minutes.push({ ts: 1786320000 + i * MIN, tokens: { m1: { input: 1 } }, costs: {} });
  }
  const out = bucketMinutes(minutes);
  eq("331 minutes widen to 5-min buckets", out.map(b => b.ts)[1] - out.map(b => b.ts)[0], 5 * MIN);
  eq("331 minutes produce 67 buckets", out.length, 67);
}

// 1440 active minutes -> 15-min buckets (96 bars)
{
  const minutes = Array.from({ length: 1440 }, (_, i) => ({
    ts: 1786320000 + i * MIN, tokens: { m1: { input: 1 } }, costs: {},
  }));
  const out = bucketMinutes(minutes);
  eq("1440 minutes widen to 15-min buckets", out.map(b => b.ts)[1] - out.map(b => b.ts)[0], 15 * MIN);
  eq("1440 minutes produce 96 buckets", out.length, 96);
}

// --- aggregation ---
// tokens/costs sum per model across merged minutes
{
  const minutes = [
    { ts: 1786320000, tokens: { m1: { input: 1, output: 2 } }, costs: { m1: 0.1 } },
    { ts: 1786320000 + MIN, tokens: { m1: { input: 10, cache_read: 5 }, m2: { output: 7 } }, costs: { m2: 0.2 } },
  ];
  // force 5-min width with enough entries? No — 2 minutes stays 1-min; test
  // aggregation by injecting a fake width via options param.
  const out = bucketMinutes(minutes, { forceWidth: 5 });
  eq("tokens aggregate per model", out[0].tokens,
     { m1: { input: 11, cache_read: 5, cache_write: 0, output: 2 },
       m2: { input: 0, cache_read: 0, cache_write: 0, output: 7 } });
  eq("costs aggregate per model", out[0].costs, { m1: 0.1, m2: 0.2 });
}

// clock alignment: buckets start at multiples of the width (xx:05, xx:10...)
{
  // 11:23 ... force 5-min width
  const minutes = [
    { ts: Date.UTC(2026, 0, 1, 11, 23) / 1000, tokens: { m1: { input: 1 } }, costs: {} },
    { ts: Date.UTC(2026, 0, 1, 11, 27) / 1000, tokens: { m1: { input: 1 } }, costs: {} },
    { ts: Date.UTC(2026, 0, 1, 11, 28) / 1000, tokens: { m1: { input: 1 } }, costs: {} },
  ];
  const out = bucketMinutes(minutes, { forceWidth: 5 });
  eq("buckets align to clock boundaries", out.map(b => new Date(b.ts * 1000).getUTCMinutes()),
     [20, 25]);
  eq("aligned buckets aggregate counts", out.map(b => b.tokens.m1.input), [1, 2]);
}

// empty input
eq("empty input yields empty output", bucketMinutes([]), []);

// renderStatsTop must populate `bucketed` BEFORE renderModelDetail consumes it
// (regression: on page load the model-detail chart rendered with the initial
// empty bucketed array — model selected, chart blank).
{
  const fn = html.match(/function renderStatsTop\(\) \{([\s\S]*?)\n\}/);
  if (!fn) {
    console.log("FAIL  renderStatsTop not found in admin.html"); failed++;
  } else {
    const body = fn[1];
    const bucketAt = body.indexOf("bucketed = ");
    const detailAt = body.indexOf("renderModelDetail(");
    if (bucketAt === -1 || detailAt === -1 || !(bucketAt < detailAt)) {
      console.log("FAIL  renderStatsTop buckets before renderModelDetail"
        + "\n  bucketAt=" + bucketAt + " detailAt=" + detailAt);
      failed++;
    } else {
      console.log("ok  renderStatsTop buckets before renderModelDetail");
    }
    // single-bucket guard: bucketWidthMin must never become Infinity
    if (/bucketed\.length \?/.test(body)) {
      console.log("FAIL  bucketWidthMin guard uses bucketed.length ? (Infinity when 1 bucket)");
      failed++;
    } else {
      console.log("ok  bucketWidthMin single-bucket guard");
    }
  }
}

if (failed) { console.error("\n" + failed + " test(s) failed"); process.exit(1); }
console.log("\nall bucketing tests passed");
