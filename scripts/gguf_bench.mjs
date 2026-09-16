/**
 * Run real prompts through a GGUF with node-llama-cpp and report timings.
 *
 * This deliberately uses node-llama-cpp rather than llama-cli: it is what the
 * DocMind Electron app loads the model with, so these numbers describe the
 * production path. It also ships an arm64 Metal prebuild, while the Homebrew
 * llama-cli on this machine is x86_64 under Rosetta.
 *
 *   node scripts/gguf_bench.mjs --model <path.gguf> --prompts <prompts.json> \
 *        [--max-tokens 256] [--context 4096] [--out results.json]
 */
import { readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";

function arg(name, fallback = null) {
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 && process.argv[index + 1] ? process.argv[index + 1] : fallback;
}

const modelPath = arg("model");
const promptsPath = arg("prompts");
const maxTokens = Number(arg("max-tokens", "256"));
const contextSize = Number(arg("context", "4096"));
const outPath = arg("out");
const modulePath = arg("module-path");

if (!modelPath || !promptsPath) {
  console.error("usage: --model <gguf> --prompts <json> [--max-tokens N] [--out FILE]");
  process.exit(2);
}

const require = createRequire(modulePath ? `${modulePath}/` : import.meta.url);
const entry = require.resolve("node-llama-cpp");
const { getLlama, LlamaChatSession } = await import(`file://${entry}`);

const prompts = JSON.parse(readFileSync(promptsPath, "utf8"));
const mib = (bytes) => Math.round((bytes / 1024 / 1024) * 10) / 10;

const t0 = performance.now();
const llama = await getLlama();
const model = await llama.loadModel({ modelPath });
const loadMs = performance.now() - t0;

const results = [];
for (const item of prompts) {
  const context = await model.createContext({ contextSize });
  const session = new LlamaChatSession({
    contextSequence: context.getSequence(),
    systemPrompt: item.system,
  });

  let firstTokenMs = null;
  let tokens = 0;
  const start = performance.now();
  const answer = await session.prompt(item.user, {
    temperature: 0,
    maxTokens,
    onTextChunk() {
      if (firstTokenMs === null) firstTokenMs = performance.now() - start;
      tokens += 1;
    },
  });
  const totalMs = performance.now() - start;
  await context.dispose();

  results.push({
    id: item.id,
    question: item.question ?? null,
    answer,
    time_to_first_token_ms: firstTokenMs === null ? null : Math.round(firstTokenMs),
    total_ms: Math.round(totalMs),
    chunks: tokens,
    chunks_per_second: totalMs > 0 ? Math.round((tokens / (totalMs / 1000)) * 100) / 100 : null,
  });
  console.error(`[gguf] ${item.id}: ${Math.round(totalMs)}ms, ${tokens} chunks`);
}

const report = {
  model_path: modelPath,
  gpu: llama.gpu,
  vram_total_mib: mib(llama.getVramState?.().total ?? 0),
  load_ms: Math.round(loadMs),
  context_size: contextSize,
  max_tokens: maxTokens,
  process_rss_mib: mib(process.memoryUsage().rss),
  prompts: results.length,
  median_chunks_per_second: median(results.map((r) => r.chunks_per_second).filter(Boolean)),
  median_ttft_ms: median(results.map((r) => r.time_to_first_token_ms).filter(Boolean)),
  results,
};

function median(values) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : Math.round(((sorted[mid - 1] + sorted[mid]) / 2) * 100) / 100;
}

await model.dispose();
const json = JSON.stringify(report, null, 2);
if (outPath) writeFileSync(outPath, json);
console.log(json);
