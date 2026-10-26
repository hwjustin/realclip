# Observation Agent 的 Summarize 模块

整个流程包含 **Research Agent 和 Observation Agent 两个 agent**。Summarize 是
Observation 内部的最后一个 module，正常执行 `realclip-d run` 或 `realclip-d describe`
时会自动运行。Research、前四个观察模块及其 prompt 的内容保持原样；编排在完整审核后
追加 Summarize，保存最终短描述。完整结构见 [Observation 模块流程](OBSERVATION_MODULES.md)。

```text
Research Agent
  ↓
Observation Agent
  baseline → guided → writer → reviewer → summarize
                                         ↓
                              处理 well 的短 description
```

配对目标是一份短描述对应一个处理 well 的多 site、多 channel 图像集合。Control 和 CP
可帮助前面的观察；短描述只保留处理图自身支持的外观，例如不能把“比对照小”直接改成“小”。

## 自动执行与结果位置

以下命令从 `<PROJECT_ROOT>/D` 运行，默认是离线 `mock`。这是用法示例，
不代表实际生成或付费调用。请使用新的输出目录。

```bash
uv run realclip-d run \
  --input examples/sample.json \
  --output runs/full-flow-demo

uv run realclip-d describe \
  --input examples/sample.json \
  --research runs/full-flow-demo/research/result.json \
  --output runs/observation-only-demo
```

`run` 执行 Research 和完整 Observation；`describe` 复用保存的 Research，再执行完整
Observation；`research` 仍只运行 Research。两种 Observation 入口都不需要额外的
Summarize 命令。

```text
runs/full-flow-demo/
  research/
  observation/
    reviewed/                 # 完整审核报告、观察证据、前四模块记录；V4
    summarize/                # 最后一个模块的输入、prompt、调用和输出记录
    result.json               # Observation 最终输出；V5
    description.txt           # 直接读取的最终短描述
    complete.json
  budget.json
  status.json
```

最终 `observation/result.json` 的 `description` 是短描述，`full_description` 是完整审核
描述；细节和证据仍在 `observation/reviewed/result.json`。完整报告与短描述分别保存，
不会把压缩文字写回前四模块结果。模型只返回纯文本，外层 JSON 的版本、路径和统计信息
由应用保存。

## 输入、筛选和输出

输入来自已完成并封存的 `observation/reviewed/`。Summarize 根据最终审核描述引用的
observations，整理与目标 CLIP 图像集合对应的可用观察、逐句上下文、限制和图像清单。
调用前检查来源完整性及目标身份。Summarize 不读取 Research 正文或 hypothesis，不
查询 CP、不重新看图，也不补回最终审核报告没有保留的发现。它使用自己的
[summarize_v1.md](../src/realclip_d/prompts/summarize_v1.md)。

图像范围由 `clip_visual_manifest` 决定。可用观察必须包含与清单对应的处理图证据；
仅由 control、未选通道或未选 crop 支持的观察不作为可写外观提供；默认五个荧光通道
不包含 Brightfield，因此默认也排除 Brightfield 专属支持。
包含混合支持的句子只保留为上下文，不能整体当作可用事实。prompt 进一步要求按具体
外观核对支持范围：一个观察引用到某个可用通道，不代表它包含的所有外观都在该通道
得到支持；无法分清时应省略。尤其不能因为细胞轮廓理论上也能在荧光图看到，就把
只在 Brightfield 中确认的轮廓移到五通道荧光 caption 中。

本版输入的 `request_version` 为 `2`，不再把未筛选的完整 `description` 直接交给模型。
`eligible_observations` 只包含最终描述已引用、确认有图像支持，且全部引用处理视图都能
对应到目标集合的观察；混有未选 Brightfield 或 crop 支持的观察整体排除，避免仅凭
“命中一个可用图像”就放行全部文字。原始文本、范围和 CP 限定保留，图像证据只提供
目标处理图。`target_scope` 分别列出观察支持的视野、集合中可用的视野和各通道视野范围，
用于避免把“提供了全部 site”误解成“在全部 site 确认了同一现象”。

`reviewed_description_context` 按最终审核句提供上下文和可用观察关联；没有可用观察的
句子不提供，部分支持的句子标记为 `partial`。prompt 要求每个写出的外观同时具有可用
观察支持、且已被最终审核描述保留，不能借上下文把被排除的另一半句子补回来，也不能
从 observation 中重新引入最终描述已删掉的细节。这里筛选的是调用前输入，不是生成后
的逐句引用检查或自动重写。

处理与对照共有、且明确包括目标处理图的外观可保留；相对比较、对照专属和 CP 专属
内容不能改成处理图的绝对外观。“比对照少”不等于“稀疏”，“没有一致增加”不等于“没有”。

观察范围同样需要保留。某个代表视野或局部区域的外观应写成“在代表视野中”或
“在部分区域中”，不能扩大为整个 well 的共同特征。多个 site 被输入或引用，也不自动
证明每个 site 都具有该外观。保留“部分细胞”“偶见”“site 间有差异”等限定，不把不同
通道的独立观察合并成同一细胞内的共现。最终文字使用自然的范围表达，省去 site 编号
和图像标签；一个清楚的范围短语可以共同限定后续几项外观，避免重复。

内容取舍由 prompt 完成：优先选择两到三个有明确支持、容易从目标图理解的外观重点，
例如整体核轮廓、染色分布或较大尺度的结构。覆盖率在对应通道中有支持时可以写，
但不固定以密度开头，也不逐通道凑齐五项。相似 well 可以得到相似 caption，不为了
拉开差异而创造表型。使用简洁自然的英文；只有已支持核身份时才直接写 nuclei，
不能把身份不明的亮点改叫细胞核。选择时也比较信息量：有明确支持的网状结构、线束
或部分明亮环缘，通常比单独写“信号广泛”“有所变化”更具体；不要为了提及某个通道
保留空泛句子，反而删掉已有的清楚结构。

默认写作偏好为 1–3 句英文、约 30–60 词，并尽量不超过 `max_words`（默认 70）。这些
要求通过 prompt 表达，不通过输出检查强制。模型直接返回 description 纯文本，**不要求
引用、证据清单、JSON 或验证状态，也没有自动修正调用**。程序只保留基本非空和英文
检查，以及调用前的来源与预算检查。若没有合适外观，prompt 要求简短说明证据限制，
不编造正常或预期表型。

## 默认图像范围与配置

完整流程根据 `observation_summary_channels` 生成处理图清单，默认配置为：

```json
{
  "observation_summary_channels": ["DNA", "RNA", "ER", "Mito", "AGP"],
  "observation_max_turns": 8
}
```

清单采用配置通道与实际原始处理图通道的交集，包含所有已提供 site；默认不包含 control
和 Brightfield。可以通过 `--config /absolute/path/config.json` 明确覆盖通道列表。
目标必须是一个身份明确的处理 well，并至少有一个被选通道；完整流程在模型调用前
检查这些条件。

当前清单是 provisional：已指定具体原尺寸来源视图，并核对来源字节身份；未来 CLIP
编码器的 resize、crop、归一化尚未确定。在目标图中已有明确支持的视野覆盖、整体细胞
轮廓、核轮廓及染色结构可以保留。不能仅因未来 resize 未定就删除这些外观，也不能仅因
来源图存在就视为已在目标集合中确认。对于极小亮点、细微
颗粒、轻微分叶或依赖局部 crop 的细节，prompt 仍要求避免尚未确认的可见性。

当前不加载本地 tokenizer，不做精确 token 或字数检查，也不因超长自动改写；文本编码器
确定后可以再单独优化训练输入长度。

## 调用、预算和恢复

Summarize 每例最多一次模型调用，没有工具、web、自动重试或修正。完整 Observation
默认八次模型请求的总额度包含这一次：前四模块预留最后一次给 Summarize；总额度至少
为五次。完整流程各模块复用同一配置与全局 `budget.json`，不会给 Summarize 另开预算。

完整审核报告与 Summarize 有各自的完成凭据。Summarize 失败时保留审核报告，整个
Observation 尚未完成；恢复不会因压缩失败而重跑成功的观察模块。已发送但未完成的
付费生成不会自动重发；不明用量会阻止后续付费调用。完整成功后，再次运行会校验并
复用结果，不再次调用模型。Replay 只复制、核对已保存的完整流程，不生成新描述。

## 单独运行：调试入口

`python -m realclip_d.summarize` 仍可单独调试已完成的完整审核报告。这是最后一个
内部 module 的调试接口，不是第三个功能 agent，也不是正常完整流程的必需步骤。
对于新完整流程，`--observation` 指向 `observation/reviewed/`。

```bash
uv run python -m realclip_d.summarize prepare \
  --observation runs/full-flow-demo/observation/reviewed \
  --channels DNA \
  --output runs/summary-debug/manifest.json

uv run python -m realclip_d.summarize check \
  --observation runs/full-flow-demo/observation/reviewed \
  --manifest runs/summary-debug/manifest.json

uv run python -m realclip_d.summarize run \
  --observation runs/full-flow-demo/observation/reviewed \
  --manifest runs/summary-debug/manifest.json \
  --output runs/summary-debug/result \
  --mode mock --fixture /absolute/path/description_fixture.txt
```

`prepare` 和 `check` 不调用模型；单独 `run --mode mock` 需要纯文本 fixture，不代表
真实生成效果。调试输出与上游完整审核目录分开；它不会更新或封存正常流程的 Observation
外层。单独真实调用需要明确 `--mode live`、预算和共享 ledger，并与 ledger 的计价及
token 预留配置一致。调试接口可通过 `--constraints` 设置长度偏好，例如：

```json
{
  "max_sentences": 3,
  "preferred_word_min": 30,
  "preferred_word_max": 60,
  "max_words": 70
}
```

复用完整报告重新 Summarize 时，应使用新的输出目录，保留之前的短描述以便对比。
这只重新运行最后一个内部模块，不修改 Research 或前四个观察模块，也不重新做图像
观察；文字更清晰或范围更准确，并不代表已经验证 CLIP 训练效果。
