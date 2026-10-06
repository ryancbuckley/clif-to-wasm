// Runs benchmark kernels in V8. Invoked by benches/codegen.rs with V8 flags
// selecting the tier, e.g.:
//   node --liftoff --no-wasm-tier-up --no-wasm-lazy-compilation run.mjs m.wasm 5 sieve:1000000
// Prints JSON: { compile_ms, kernels: { name: { ms: [...], result } } }
import fs from 'fs';

const [file, repsArg, ...specs] = process.argv.slice(2);
const reps = Number(repsArg);
const bytes = fs.readFileSync(file);

// V8 caches compiled modules by their bytes, so make every copy unique by
// appending a custom section.
const unique = (i) => {
  const name = new TextEncoder().encode('bench' + String(i).padStart(6, '0'));
  const out = new Uint8Array(bytes.length + 3 + name.length);
  out.set(bytes);
  out.set([0, 1 + name.length, name.length], bytes.length);
  out.set(name, bytes.length + 3);
  return out;
};
let module = new WebAssembly.Module(unique(0)); // warm up
const compileRuns = 20;
const copies = Array.from({ length: compileRuns }, (_, i) => unique(i + 1));
const t0 = performance.now();
for (const c of copies) new WebAssembly.Module(c);
const compile_ms = (performance.now() - t0) / compileRuns;

const kernels = {};
for (const spec of specs) {
  const [name, n] = spec.split(':');
  const ms = [];
  let result;
  for (let r = 0; r < reps; r++) {
    const instance = new WebAssembly.Instance(module, {});
    const f = instance.exports['k_' + name];
    const s = performance.now();
    result = f(Number(n));
    ms.push(performance.now() - s);
  }
  kernels[name] = { ms, result: String(result) };
}
console.log(JSON.stringify({ compile_ms, kernels }));
