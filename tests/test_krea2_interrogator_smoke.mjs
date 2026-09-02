import fs from "fs";
import path from "path";
import os from "os";
import { fileURLToPath } from "url";
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.resolve(__dirname, "..", "web");
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "sf_interrogator_"));

let failures=[];
function check(n,c){ if(c) console.log("PASS:",n); else{ failures.push(n); console.log("FAIL:",n);}}

// stub app + graph
const fakeGraph = { _nodes: [] };
globalThis.app = { graph: fakeGraph, registerExtension(ext){
  globalThis.__EXT__=ext;
}};
fs.writeFileSync(path.join(tmp,"stub_app.mjs"), `export const app = globalThis.app;`);

// mock modules required by krea2_interrogator.js
fs.writeFileSync(path.join(tmp,"sf_krea2_presets.mjs"), `
export async function fetchPresets(){ return {presets:{}}; }
export function addManageButton(){}
export function setPresetOptions(){}
export async function reloadNodes(){ return {presets:{}}; }
export function presetsChangedEvent(k){ return "sfnodes."+k+"-presets-changed"; }
export function nodesOfClass(){ return []; }
`);
fs.writeFileSync(path.join(tmp,"sf_popup.mjs"), `export function attachPopupDismiss(){}
export function clampToViewport(){}`);
fs.writeFileSync(path.join(tmp,"sf_common.mjs"), `export function el(){return{}}
export function injectCSSOnce(){}`);

// transform source
let src = fs.readFileSync(path.join(WEB,"krea2_interrogator.js"),"utf-8");
src = src.replace('import { app } from "/scripts/app.js";','import { app } from "./stub_app.mjs";');
src = src.replace('from "./sf_krea2_presets.js"','from "./sf_krea2_presets.mjs"');
src = src.replace('from "./sf_popup.js"','from "./sf_popup.mjs"');
src = src.replace('from "./sf_common.js"','from "./sf_common.mjs"');
fs.writeFileSync(path.join(tmp,"mod.mjs"), src);

// need to provide document stub for setup
globalThis.document = { addEventListener(){}, createElement:()=>({style:{},setProperty(){}}), getElementById:()=>null, body:{appendChild(){}}, head:{appendChild(){}} };
globalThis.CustomEvent = class {constructor(t,o){this.type=t;this.detail=o?.detail}};

await import(path.join(tmp,"mod.mjs"));
const ext = globalThis.__EXT__;
if(!ext) { console.log("FAIL: extension not registered"); process.exit(1); }

// Helper to make node
function makeNode(widgets){
  return { comfyClass:"SFImageInterrogator", widgets: widgets.map(w=>({name:w.name, type:w.type, value:w.value, options:w.options||{}})), setDirtyCanvas(){this._dirty=true;}, properties:{} };
}

// Test 1: old workflow pollution -> control numeric 1 should heal to fixed
{
  fakeGraph._nodes = [];
  const n = makeNode([
    {name:"preset", type:"combo", value:"default"},
    {name:"prompt", type:"STRING", value:"hello"},
    {name:"max_length", type:"INT", value:256},
    {name:"do_sample", type:"BOOLEAN", value:true},
    {name:"temperature", type:"FLOAT", value:0.7},
    {name:"top_k", type:"INT", value:64},
    {name:"top_p", type:"FLOAT", value:0.95},
    {name:"repetition_penalty", type:"FLOAT", value:1.05},
    {name:"seed", type:"INT", value:42},
    {name:"control_after_generate", type:"combo", value:1}, // polluted numeric from vision 1.0
    {name:"vision_megapixels", type:"FLOAT", value:1.0},
    {name:"thinking", type:"BOOLEAN", value:false},
  ]);
  fakeGraph._nodes.push(n);
  ext.afterConfigureGraph();
  check("control numeric 1 heals to fixed", n.widgets.find(w=>w.name==="control_after_generate").value==="fixed");
  check("seed stays 42", n.widgets.find(w=>w.name==="seed").value===42);
  check("vision stays 1.0", n.widgets.find(w=>w.name==="vision_megapixels").value===1.0);
}

// Test 2: control string pollution fallback
{
  fakeGraph._nodes = [];
  const n = makeNode([
    {name:"seed", type:"INT", value:5},
    {name:"control_after_generate", type:"combo", value:"invalid"},
    {name:"vision_megapixels", type:"FLOAT", value:1.0},
    {name:"thinking", type:"BOOLEAN", value:false},
  ]);
  fakeGraph._nodes.push(n);
  ext.afterConfigureGraph();
  check("control invalid string heals to fixed", n.widgets.find(w=>w.name==="control_after_generate").value==="fixed");
}

// Test 3: control correct value preserved
{
  fakeGraph._nodes = [];
  const n = makeNode([
    {name:"seed", type:"INT", value:0},
    {name:"control_after_generate", type:"combo", value:"increment"},
    {name:"vision_megapixels", type:"FLOAT", value:2.0},
    {name:"thinking", type:"BOOLEAN", value:false},
  ]);
  fakeGraph._nodes.push(n);
  ext.afterConfigureGraph();
  check("control increment preserved", n.widgets.find(w=>w.name==="control_after_generate").value==="increment");
}

// Test 4: seed string pollution heals to 0
{
  fakeGraph._nodes = [];
  const n = makeNode([
    {name:"seed", type:"INT", value:"fixed"},
    {name:"control_after_generate", type:"combo", value:"fixed"},
    {name:"vision_megapixels", type:"FLOAT", value:1.0},
    {name:"thinking", type:"BOOLEAN", value:false},
  ]);
  fakeGraph._nodes.push(n);
  ext.afterConfigureGraph();
  check("seed string pollution heals to 0", n.widgets.find(w=>w.name==="seed").value===0);
}

// Test 5: vision string pollution heals to 1.0
{
  fakeGraph._nodes = [];
  const n = makeNode([
    {name:"seed", type:"INT", value:10},
    {name:"control_after_generate", type:"combo", value:"fixed"},
    {name:"vision_megapixels", type:"FLOAT", value:"not-a-number"},
    {name:"thinking", type:"BOOLEAN", value:false},
  ]);
  fakeGraph._nodes.push(n);
  ext.afterConfigureGraph();
  check("vision string pollution heals to 1.0", n.widgets.find(w=>w.name==="vision_megapixels").value===1.0);
}

// Test 6: thinking numeric pollution heals to false
{
  fakeGraph._nodes = [];
  const n = makeNode([
    {name:"seed", type:"INT", value:0},
    {name:"control_after_generate", type:"combo", value:"fixed"},
    {name:"vision_megapixels", type:"FLOAT", value:1.0},
    {name:"thinking", type:"BOOLEAN", value:1},
  ]);
  fakeGraph._nodes.push(n);
  ext.afterConfigureGraph();
  const tv = n.widgets.find(w=>w.name==="thinking").value;
  check("thinking numeric 1 heals to false", tv===false);
}

// Test 7: thinking string pollution
{
  fakeGraph._nodes = [];
  const n = makeNode([
    {name:"seed", type:"INT", value:0},
    {name:"control_after_generate", type:"combo", value:"fixed"},
    {name:"vision_megapixels", type:"FLOAT", value:1.0},
    {name:"thinking", type:"BOOLEAN", value:"hello"},
  ]);
  fakeGraph._nodes.push(n);
  ext.afterConfigureGraph();
  check("thinking random string heals to false", n.widgets.find(w=>w.name==="thinking").value===false);
}

// Test 8: old 11-value workflow without control (simulate missing widget) should not crash
{
  fakeGraph._nodes = [];
  const n = makeNode([
    {name:"seed", type:"INT", value:42},
    {name:"vision_megapixels", type:"FLOAT", value:1.0},
    {name:"thinking", type:"BOOLEAN", value:false},
  ]);
  // no control widget at all (old frontend without implicit control)
  fakeGraph._nodes.push(n);
  try { ext.afterConfigureGraph(); check("missing control widget no crash", true); }
  catch(e){ check("missing control widget no crash", false); }
}

if(failures.length){ console.log("\nFAIL:",failures.join(", ")); process.exit(1); }
console.log("\nALL PASS");
