"""LoRA 文件缓存 + 内存模式修剪(SFLoraPlot 使用;语义与 SFLoraStack 一致)。

三种内存模式(节点状态里 cacheMode 携带):
  "last"(默认) = ComfyUI 对齐,只留最近使用的文件;
  "all"  = 保留整个当前行集合(重跑最快,大栈可达 GB 级);
  "none" = 什么都不留,每次 run 重读(内存最低)。

"last"/"none" 下本次 run 每条目应用完下一个时立即释放(note_applied),
峰值内存保持在几个文件而不是整个集合(跨 run 保留条目一直随行到结束)。
纯 dict 操作,无 comfy 依赖,可独立单测。
"""


class LoraCache:
    def __init__(self):
        self._data = {}
        self._last_path = None

    def get(self, path):
        """命中返回缓存条目,未命中返回 None(调用方负责加载并 store)。"""
        return self._data.get(path)

    def store(self, path, value):
        self._data[path] = value

    def note_applied(self, path, mode, last_this_run):
        """应用某文件后调用,控制本 run 峰值:mode != "all" 时逐出本 run
        加载的上一行(暖文件刚用毕即释放)。返回本 run 最近加载的路径。
        跨 run 保留条目(self._last_path)刻意不在这里碰——它后面可能还被
        复用。"""
        if mode != "all":
            if last_this_run is not None and last_this_run != path:
                self._data.pop(last_this_run, None)
            return path
        return None

    def trim(self, mode, used_paths, last_this_run):
        """run 结束后按模式修剪。used_paths:本 run 实际使用(含刻意
        无操作的强度 0)的文件路径集合;last_this_run:本 run 最近加载的
        路径(note_applied 的返回值)。"""
        if mode == "none":
            self._data.clear()
            self._last_path = None
        elif mode == "all":
            # 释放用户删掉的 LoRA 条目,让内存跟随节点。
            for path in list(self._data):
                if path not in used_paths:
                    del self._data[path]
        else:  # "last":最多一个条目活过本次 run——本次 run 最近一次加载;
            # 本次没加载任何东西时,先前保留的文件只在它仍是集合的一部分时
            # 存活(强度 0 的行也算)——清空的行集合真的释放它。
            keep = last_this_run
            if keep is None and self._last_path in used_paths:
                keep = self._last_path
            for path in list(self._data):
                if path != keep:
                    del self._data[path]
            self._last_path = keep
