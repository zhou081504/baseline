# 来源与许可

最终分类器更新为 [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)。原方法来源和适配边界保留如下；如历史文字提及下游中文BERT，以当前 PROTOCOL.md 为准。

# 第三方来源与许可

`table_baselines/cogltx.py` 参考 Sleepychord/CogLTX 的 memory replay、reasoner 和训练主循环实现；原项目在 `Selection_Retrieval/CogLTX/upstream/LICENSE`：Copyright (c) 2020 Ming Ding，MIT。

`table_baselines/carrot.py` 参考 wang0702/CARROT 的 MCTS 和 node 机制；原项目在 `Selection_Retrieval/CARROT/upstream/LICENSE`：Copyright (c) 2025 Ziting Wang，MIT。

LongLLMLingua 直接导入既有 vendored Microsoft 代码，原版权声明为 Copyright (c) 2023-2025 Microsoft，MIT；未改写该目录。

上述 MIT 许可条款：

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

预训练模型遵循各自模型仓库的许可，下载工具保留可获取的 LICENSE 和 README.md，记录精确 revision。
中文 Wikipedia 句子对保留页面归属及 revision 信息，遵循来源页面的 CC BY-SA 条款；详见数据目录 `provenance.json`。
