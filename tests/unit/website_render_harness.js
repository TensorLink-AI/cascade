// Render a dashboard page's inline scripts against fixture JSON in a minimal
// DOM shim (no browser, no network) and print every element's rendered
// innerHTML/textContent by id. Driven by tests/unit/test_website_render.py.
//
//   node website_render_harness.js <page.html> <fixtures.json> <hostname> <calls.json>
//
// fixtures.json maps a bucket path ("receipts/index.json") to its JSON body;
// any other path answers 404 (the pages' "missing is final" branch). After
// the fetches settle, each call in calls.json (e.g. "renderAll()") is
// evaluated in the page's global scope so a render bug THROWS here instead
// of being swallowed by a fetch .catch().
"use strict";
const fs = require("fs");
const vm = require("vm");

const [page, fixturesPath, hostname, callsPath] = process.argv.slice(2);
const html = fs.readFileSync(page, "utf8");
const fixtures = JSON.parse(fs.readFileSync(fixturesPath, "utf8"));
const calls = JSON.parse(fs.readFileSync(callsPath, "utf8"));

const els = {};
function makeEl(id) {
  const el = {
    id, innerHTML: "", textContent: "", hidden: false, className: "", title: "", value: "",
    style: {}, dataset: {}, children: [], clientWidth: 1000, clientHeight: 420,
    offsetWidth: 160, offsetHeight: 20, offsetTop: 0, offsetLeft: 0,
    classList: { toggle() {}, add() {}, remove() {}, contains() { return false; } },
    addEventListener() {}, removeEventListener() {}, setAttribute() {}, getAttribute() { return null; },
    appendChild() {}, removeChild() {}, contains() { return false; },
    querySelector() { return makeEl(""); }, querySelectorAll() { return []; },
    getBoundingClientRect() { return { left: 0, top: 0, width: 1000, height: 420 }; },
    select() {},
  };
  return el;
}
// ids that exist in the markup (anything else is null, as in a browser)
const ids = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1]));
// data-* buttons the wiring binds by attribute
const document = {
  title: "cascade", hidden: false,
  documentElement: { dataset: {} },
  getElementById(id) {
    if (!ids.has(id)) return null;
    return els[id] || (els[id] = makeEl(id));
  },
  querySelector() { return makeEl(""); },
  querySelectorAll() { return []; },
  addEventListener() {}, createElement() { return makeEl(""); },
  body: makeEl("body"), execCommand() {},
};
function respond(path) {
  const clean = path.replace(/^https?:\/\/[^/]+\/[^/]+\//, "").replace(/^\/+/, "");
  if (Object.prototype.hasOwnProperty.call(fixtures, clean)) {
    const body = fixtures[clean];
    return Promise.resolve({ status: 200, ok: true, headers: { get() { return null; } },
                             json() { return Promise.resolve(JSON.parse(JSON.stringify(body))); } });
  }
  return Promise.resolve({ status: 404, ok: false, headers: { get() { return null; } },
                           json() { return Promise.reject(new Error("404")); } });
}
const fetched = [];
const ctx = {
  document, window: {}, console,
  location: { hostname, pathname: "/index.html" },
  localStorage: { getItem() { return null; }, setItem() {} },
  navigator: {},
  fetch(url) { fetched.push(url); return respond(url); },
  AbortController: class { constructor() { this.signal = {}; } abort() {} },
  setTimeout(fn) { return 0; }, clearTimeout() {}, setInterval() { return 0; }, clearInterval() {},
  Promise, Date, Math, JSON, Number, String, Object, Array, isFinite, isNaN, parseInt, parseFloat,
};
ctx.window = ctx;
ctx.addEventListener = function () {};
ctx.setImmediate = setImmediate;
vm.createContext(ctx);

const scripts = [...html.matchAll(/<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/g)].map((m) => m[1]);
for (const s of scripts) vm.runInContext(s, ctx, { filename: page });

function settle(n) { let p = Promise.resolve(); for (let i = 0; i < n; i++) p = p.then(() => new Promise((r) => setImmediate(r))); return p; }

settle(40).then(() => {
  const errors = [];
  for (const c of calls) {
    try { vm.runInContext(c, ctx); } catch (e) { errors.push(`${c}: ${e && e.stack || e}`); }
  }
  return settle(10).then(() => {
    const out = {};
    for (const [id, el] of Object.entries(els)) out[id] = { html: el.innerHTML, text: el.textContent, hidden: el.hidden, cls: el.className };
    process.stdout.write(JSON.stringify({ errors, els: out, fetched }));
  });
}).catch((e) => { process.stdout.write(JSON.stringify({ errors: [String(e && e.stack || e)], els: {}, fetched })); });
