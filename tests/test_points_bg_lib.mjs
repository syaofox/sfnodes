// sf_points_bg_lib 纯逻辑测试（Node 直接运行：node tests/test_points_bg_lib.mjs）
// 覆盖：describeSource（preview/videoEl/widget/file 优先级）、
// parseCropExpandState（JSON 串/对象/坏值）、resolveBgSource 链式解析/循环保护、
// makeGraphApi（links Map 与对象表、IMAGE 槽优先、`*` 回退）
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const here = path.dirname(new URL(import.meta.url).pathname);
const tmpMjs = path.join(os.tmpdir(), "sf_points_bg_lib_test.mjs");
fs.copyFileSync(path.join(here, "..", "web", "sf_points_bg_lib.js"), tmpMjs);

const failures = [];
function check(name, cond) {
    if (cond) console.log("PASS:", name);
    else { failures.push(name); console.log("FAIL:", name); }
}

const L = await import(pathToFileURL(tmpMjs).href);

// ── fake graph helpers ──
const nodes = {};
function mkNode(id, opts = {}) {
    const n = { id, inputs: [], widgets: [], properties: {}, ...opts };
    nodes[id] = n;
    return n;
}
function inp(name, type, link) { return { name, type, link }; }
function mkGraph(links) {
    return {
        links,  // object table {id: {origin_id}}
        getNodeById: (id) => nodes[id] ?? null,
    };
}
function reset() { for (const k of Object.keys(nodes)) delete nodes[k]; }

// ── describeSource ──
check("空返回 null", L.describeSource(null) === null);
check("无源返回 null", L.describeSource(mkNode(1)) === null);

const imgNode = mkNode(2, { widgets: [{ name: "image", value: "sub/pic.png" }] });
check("image widget", JSON.stringify(L.describeSource(imgNode)) ===
    JSON.stringify({ kind: "widget", widget: "image", value: "sub/pic.png" }));

const vidNode = mkNode(3, { widgets: [{ name: "video", value: "clip.mp4" }] });
check("video widget", L.describeSource(vidNode).kind === "widget" && L.describeSource(vidNode).widget === "video");

const vhsNode = mkNode(4, { widgets: [{ name: "videopreview", videoEl: { src: "/view?x" } }] });
check("videoEl", L.describeSource(vhsNode).kind === "videoEl");

const prevNode = mkNode(5, { imgs: [{ src: "/view?preview" }], widgets: [{ name: "image", value: "x.png" }] });
check("preview 优先于 widget", L.describeSource(prevNode).kind === "preview"
    && L.describeSource(prevNode).url === "/view?preview");
check("imgs 字符串形态", L.describeSource({ imgs: ["/a.png"], widgets: [] }).url === "/a.png");

const cropNode = mkNode(6, { properties: { sfCropExpandState: JSON.stringify({ src_path: "sfnodes_crop/a.png" }) } });
check("crop file", L.describeSource(cropNode).kind === "file"
    && L.describeSource(cropNode).path === "sfnodes_crop/a.png");

// ── parseCropExpandState ──
check("state 对象", L.parseCropExpandState({ properties: { sfCropExpandState: { src_path: "a" } } }).src_path === "a");
check("state 坏 JSON", L.parseCropExpandState({ properties: { sfCropExpandState: "{bad" } }) === null);
check("state 缺失", L.parseCropExpandState({ properties: {} }) === null);

// ── resolveBgSource 链式 ──
reset();
const editor = mkNode(10, { inputs: [inp("bg_image", "IMAGE", 100)] });
const scale = mkNode(11, { inputs: [inp("image", "IMAGE", 101)] });
const sw = mkNode(12, { inputs: [inp("any_01", "IMAGE", 102), inp("any_02", "IMAGE", null)] });
const batch = mkNode(13, { inputs: [inp("image_1", "IMAGE", 103), inp("image_2", "IMAGE", null)] });
const crop = mkNode(14, { properties: { sfCropExpandState: JSON.stringify({ src_path: "src.png" }) } });
const g = mkGraph({
    100: { origin_id: 11 },
    101: { origin_id: 12 },
    102: { origin_id: 13 },
    103: { origin_id: 14 },
});
const api = L.makeGraphApi(g);
check("链式解析到裁剪源", L.resolveBgSource(editor, api)?.kind === "file"
    && L.resolveBgSource(editor, api)?.path === "src.png");

// preview 在链中优先
reset();
const crop2 = mkNode(20, { imgs: [{ src: "/p.png" }] });
const e2 = mkNode(21, { inputs: [inp("bg_image", "IMAGE", 200)] });
const g2 = mkGraph({ 200: { origin_id: 20 } });
check("已执行预览优先", L.resolveBgSource(e2, L.makeGraphApi(g2))?.kind === "preview");

// 循环保护
reset();
const a = mkNode(30, { inputs: [inp("bg_image", "IMAGE", 300)] });
const b = mkNode(31, { inputs: [inp("image", "IMAGE", 301)] });
const g3 = mkGraph({ 300: { origin_id: 31 }, 301: { origin_id: 30 } });
check("循环返回 null", L.resolveBgSource(a, L.makeGraphApi(g3)) === null);

// 未接线
reset();
const solo = mkNode(40, { inputs: [inp("bg_image", "IMAGE", null)] });
check("未接线 null", L.resolveBgSource(solo, L.makeGraphApi(mkGraph({}))) === null);

// ── links Map 形态 + `*` 回退 ──
reset();
const mapEditor = mkNode(50, { inputs: [inp("bg_image", "IMAGE", 500)] });
const anyNode = mkNode(51, { inputs: [inp("on_false", "*", 501)] });
const loadNode = mkNode(52, { widgets: [{ name: "image", value: "y.png" }] });
const mapLinks = new Map([[500, { origin_id: 51 }], [501, { origin_id: 52 }]]);
const mapGraph = { links: mapLinks, getNodeById: (id) => nodes[id] ?? null };
check("Map links + * 回退", L.resolveBgSource(mapEditor, L.makeGraphApi(mapGraph))?.value === "y.png");

if (failures.length) {
    console.log("FAILED:", failures.length, failures);
    process.exit(1);
}
console.log("ALL PASS");
