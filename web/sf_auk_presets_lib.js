// SF AuK Generate / Edit 官方提示词模板数据（快速填入 instruction 用）
//
// 文本来源：Tencent-Hunyuan/AuK（MIT）docs/COOKBOOK.md 与 src/auk/infer/infer_gradio.py
// 官方演示原句；中文标题与分组为本包整理。模板中的 {花括号} 为占位符，填入后按需替换。
//
// 纯数据 + 纯函数模块：无 app/ComfyUI 依赖，可拷贝为 .mjs 直接测试
// （tests/test_auk_presets_lib.mjs）。控件挂载与填入逻辑在 sf_auk_generate.js。
//
// scope（适用场景，组级默认、条目级可覆盖）：
//   generate = SFAuKGenerateEdit（单段 ≤30s，全部模板可用）
//   process  = SFAuKLongSpeech 长音频处理模式（逐块套用；排除 TTS、内容编辑/歌词、
//              多人分离/按内容提取，以及"开头/结尾加声"这类位置型条目）

export const SCOPE_GENERATE = "generate";
export const SCOPE_PROCESS = "process";

export const AUK_PRESET_GROUPS = [
    {
        name: "1. 参考音色 TTS",
        scopes: ["generate"],
        items: [
            { label: "EN & CN", text: `Say the following with the same voice: "{text}"` },
            { label: "官方演示原句", text: `Say the following with the same voice: 'Ladies and gentlemen, it's an honor to have the opportunity to address such a distinguished audience'` },
        ],
    },
    {
        name: "2. 声音描述 TTS",
        scopes: ["generate"],
        items: [
            { label: "EN", text: `Generate speech based on the following description: "{voice description}". The content to speak is: "{text}".` },
            { label: "CN", text: `请基于下面的描述: "{声音描述}",生成语音内容"{文本}".` },
            { label: "官方演示原句", text: `Say the following in the voice described here: “一位雄才大略、性格复杂的乱世枭雄，以略显沙哑却极有穿透力的中年男声说话。语气自信、果断，带着审视人心的敏锐感。讲话时节奏变化明显，可以先压低声音缓缓铺垫，再突然加重关键字。既有豪迈，也隐约带着危险与猜疑”, and say: “宁可我负天下人，休教天下人负我。”` },
        ],
    },
    {
        name: "3. 语音内容编辑（替换、增添、删除）",
        scopes: ["generate"],
        items: [
            { label: "Replace · EN", text: `Replace '{original}' with '{new}'.` },
            { label: "Replace · CN", text: `把‘{原文}’改成‘{新文}’` },
            { label: "Insert before · EN", text: `Add '{text}' before '{anchor}'.` },
            { label: "Insert before · CN", text: `在‘{锚点}’前面加上‘{内容}’` },
            { label: "Insert after · EN", text: `Add '{text}' after '{anchor}'.` },
            { label: "Insert after · CN", text: `在‘{锚点}’后面加上‘{内容}’` },
            { label: "Remove · EN", text: `Remove '{text}'.` },
            { label: "Remove · CN", text: `删掉‘{内容}’` },
            { label: "Remove with anchor · EN", text: `Remove '{text}' before/after '{anchor}'.` },
            { label: "Remove with anchor · CN", text: `删掉‘{锚点}’前/后面的‘{内容}’` },
            { label: "官方演示原句", text: `Replace 'but accepting what we cannot have' with 'and living well with dreams unmet'.` },
        ],
    },
    {
        name: "4. 歌词编辑",
        scopes: ["generate"],
        items: [
            { label: "EN", text: `Change "{original lyrics}" to "{new lyrics}" in the vocal recording.` },
            { label: "CN", text: `把这段歌词中的“{原歌词}”改成“{新歌词}”。` },
            { label: "官方演示原句", text: `Replace “rear view” with “like you” in the lyrics` },
        ],
    },
    {
        name: "5. 音高调整",
        scopes: ["generate", "process"],
        items: [
            { label: "Raise · EN", text: `Raise the pitch by {1/2/3} semitones.` },
            { label: "Raise · CN", text: `将音调升高{1/2/3}个半音。` },
            { label: "Lower · EN", text: `Lower the pitch by {1/2/3} semitones.` },
            { label: "Lower · CN", text: `将音调降低{1/2/3}个半音。` },
            { label: "官方演示原句", text: `将音调降低3个半音。` },
        ],
    },
    {
        name: "6. 语速调整",
        scopes: ["generate", "process"],
        items: [
            { label: "EN", text: `Adjust the speech speed to {0.5/0.75/1.25/1.5/2.0}x.` },
            { label: "CN", text: `将语速调整为{0.5/0.75/1.25/1.5/2.0}倍。` },
            { label: "官方演示原句", text: `将语速调整为0.5倍。` },
        ],
    },
    {
        name: "7. 音量调整",
        scopes: ["generate", "process"],
        items: [
            { label: "Increase · EN", text: `Increase the volume by {5/10/15} dB.` },
            { label: "Increase · CN", text: `将音量升高{5/10/15}分贝。` },
            { label: "Decrease · EN", text: `Decrease the volume by {5/10/15} dB.` },
            { label: "Decrease · CN", text: `将音量降低{5/10/15}分贝。` },
            { label: "官方演示原句", text: `将音量降低15分贝。` },
        ],
    },
    {
        name: "8. 情绪转换",
        scopes: ["generate", "process"],
        items: [
            { label: "EN", text: `Change the emotion to {happy/angry/sad/fearful/surprised/disgusted/calm/excited}.` },
            { label: "CN", text: `将情感转变为{开心/愤怒/悲伤/恐惧/惊讶/厌恶/平静/兴奋}。` },
            { label: "官方演示原句", text: `Say this in a happy tone` },
        ],
    },
    {
        name: "9. 音色转换",
        scopes: ["generate", "process"],
        items: [
            { label: "EN", text: `Keep the spoken content unchanged and change the timbre to: "{description}".` },
            { label: "CN", text: `请将这段音频的音色修改为符合以下描述的声音：“{音色描述}”。` },
            { label: "官方演示原句", text: `Keep the words and change the timbre to: “这位说话人的声音低沉而浑厚，语速平稳，吐字清晰。他的说话风格沉稳而富有思考，带有平静的反思特质。”` },
        ],
    },
    {
        name: "10. 去口音",
        scopes: ["generate", "process"],
        items: [
            { label: "EN", text: `Remove the regional accent while preserving the speaker's voice and content.` },
            { label: "CN", text: `请去掉这段语音里的方言口音，保持说话人音色一致。` },
            { label: "官方演示原句", text: `请去掉这段语音里的方言口音，保持说话人音色一致。` },
        ],
    },
    {
        name: "11. 非语言声音编辑",
        scopes: ["generate", "process"],
        items: [
            { label: "Remove · EN", text: `Remove all {breaths/laughs/coughs/etc.} from the audio.` },
            { label: "Remove · CN", text: `删除音频中所有的{换气声/笑声/咳嗽声等}。` },
            { label: "Add · EN", text: `Add a {sound} at the {beginning/end} of the speech.`, scopes: ["generate"] },
            { label: "Add · CN", text: `在语音{开头/结尾}增加{声音}。`, scopes: ["generate"] },
            { label: "官方演示原句", text: `Add a breath before “We tested”`, scopes: ["generate"] },
        ],
    },
    {
        name: "12. 耳语与正常语音互转",
        scopes: ["generate", "process"],
        items: [
            { label: "To whisper · EN", text: `Convert this speech into a soft whisper while preserving the speaker and content.` },
            { label: "To whisper · CN", text: `用小声耳语的方式把这段话说出来。` },
            { label: "From whisper · EN", text: `Convert this whispered speech into a normal speaking voice while preserving the speaker and content.` },
            { label: "From whisper · CN", text: `把这段耳语转换成正常说话的声音。` },
            { label: "官方演示原句", text: `Turn this into a whisper` },
        ],
    },
    {
        name: "13. 语音增强（降噪、去混响、修复）",
        scopes: ["generate", "process"],
        items: [
            { label: "Denoise · EN", text: `Remove only the background noise, preserve everything else, and output audio of the same length.` },
            { label: "Denoise · CN", text: `请只去除背景噪声，保留其他内容，输出等长结果。` },
            { label: "Dereverberate · EN", text: `Remove only the room reverberation, preserve everything else, and output audio of the same length.` },
            { label: "Dereverberate · CN", text: `请只去除房间混响，保留其他内容，输出等长结果。` },
            { label: "Enhance · EN", text: `Preserve all speakers, remove noise and reverberation, and output clean speech of the same length.` },
            { label: "Enhance · CN", text: `请保留所有说话人，去除噪声和混响，输出等长的纯净语音。` },
            { label: "Restoration · EN", text: `Repair the {telephone effect/muffling/clipping/dropouts} and restore natural, clear speech.` },
            { label: "Restoration · CN", text: `请修复这段音频的{电话感/闷声/削波/丢包}，恢复自然清晰的人声。` },
            { label: "官方演示原句", text: `Remove the background noise and make the voice cleaner` },
        ],
    },
    {
        name: "14. 多人语音分离",
        scopes: ["generate"],
        items: [
            { label: "EN", text: `Keep only the {first/second/etc.} speaker to start talking and remove all other speakers.` },
            { label: "CN", text: `只保留第{序号}个开始说话的人，去掉其余说话人。` },
            { label: "官方演示原句", text: `Keep only the speaker who says “get what”` },
        ],
    },
    {
        name: "15. 音乐人声分离",
        scopes: ["generate", "process"],
        items: [
            { label: "Singing only · EN", text: `Keep only the singing voice and remove everything else.` },
            { label: "Singing only · CN", text: `请只保留歌声，其余声音都去掉。` },
            { label: "All human voices · EN", text: `Keep all human voices, including speech and singing, and remove everything else.` },
            { label: "All human voices · CN", text: `请保留所有人声，包括说话和歌唱，其余声音都去掉。` },
            { label: "官方演示原句", text: `Extract the vocals and remove the accompaniment` },
        ],
    },
    {
        name: "16. 按说话内容提取目标说话人",
        scopes: ["generate"],
        items: [
            { label: "EN", text: `Keep only the speaker who says "{content}" and remove all other speakers.` },
            { label: "CN", text: `请只保留说“{内容}”的人，去掉其他说话人。` },
            { label: "官方演示原句", text: `Keep only the speaker who says “警队规矩”` },
        ],
    },
    {
        name: "附：官方音质改善演示",
        scopes: ["generate", "process"],
        items: [
            { label: "官方演示原句", text: `Improve the audio quality and make it clearer` },
        ],
    },
];

export const DEFAULT_GROUP = AUK_PRESET_GROUPS[0].name;
export const DEFAULT_TEMPLATE = AUK_PRESET_GROUPS[0].items[0].label;

function itemScopes(group, item) {
    return item.scopes ?? group.scopes ?? [SCOPE_GENERATE];
}

export function groupNames(scope = SCOPE_GENERATE) {
    return AUK_PRESET_GROUPS
        .filter((group) => group.items.some((item) => itemScopes(group, item).includes(scope)))
        .map((group) => group.name);
}

export function itemsOf(groupName, scope = SCOPE_GENERATE) {
    const group = AUK_PRESET_GROUPS.find((entry) => entry.name === groupName);
    if (!group) return [];
    return group.items.filter((item) => itemScopes(group, item).includes(scope));
}

export function templateText(groupName, label) {
    const item = itemsOf(groupName).find((entry) => entry.label === label);
    return item ? item.text : "";
}
