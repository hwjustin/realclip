# 当前模块的任务背景审查

检查日期：2026-10-22。结论针对当前实际运行路径，并核对了保存的真实请求；不以目录里存在某个历史 prompt 推断它正在运行。

## 结论

**Cell Painting 背景已明确；处理 well 做 CLIP 的终端目标尚未明确。** 这两个结论不能混为一谈。所有模块“应该知道”的目标是设计要求，不是对当前实现状态的描述。

| 活动模块 | Cell Painting | well/site/channel/control 语义 | 处理 well 图像集合与短 caption 对齐 | 当前职责 |
| --- | --- | --- | --- | --- |
| Research | 明确 | 部分：指导 treated/control 比较，未收到具体图像清单 | 未明确 | 研究证据、形态假设及观察指导 |
| Baseline | 明确 | 明确：原图、site 异质性、配准、well/replicate 区别 | 未明确 | 不看假设与 CP 数值，开放图像观察 |
| Guided | 明确 | 明确：原图/裁图、比较、CP 尺度与解释边界 | 未明确 | 保留基础观察，补充复看与证据限定 |
| Writer | 明确 | 部分：保留角色、通道、site 范围；不接收图像清单 | 未明确 | 写完整 well comparison，允许 control-only 和比较句 |
| Reviewer | 明确 | 明确：局部/全景、不同 site/cell、跨通道关系 | 未明确 | 对完整观察报告逐句 keep/revise/delete |
| Summarize | 尚未实现 | 本组草案明确设计 | 本组草案明确设计 | 将最终报告投影并压缩为处理 well 的短 caption |

这不是说上游不了解 Cell Painting。缺少的是跨模块的终端任务声明，以及确定哪些图真正进入 CLIP 的输入清单。

## 实际加载路径

### Research

`agent_runtime._prompt()` 默认版本为 v3。Research Agent 用 `_prompt("research")`，实际 instructions 是 `research_v3.md`。`research_output_format_v3.md` 以 developer message 加入请求，确实进入模型上下文。输出 schema 名称是 `ResearchResultV4`，不能据此把活动 prompt 称为 v4。

证据：

- [默认 prompt 版本与消息组装](../../src/realclip_d/agent_runtime.py#L426)。
- [Research instructions、输出格式及运行](../../src/realclip_d/agent_runtime.py#L605)。
- [活动 Research prompt](../../src/realclip_d/prompts/research_v3.md#L5)。
- [活动 Research 输出格式](../../src/realclip_d/prompts/research_output_format_v3.md#L1)。

Research prompt 第 8–11 行明确写到 “observable in Cell Painting images”，并说明假设用于指导 Observation 生成 morphology descriptions；第 43–49 行要求比较 treated/control。未说明 CLIP、contrastive learning、一个处理 well 的多图样本或短 caption 目标。

Research payload 只传入 sample ID、化合物、known MOA 和实验 context。`ExperimentContext` 当前定义 cell_line、dose、treatment_duration、notes，没有 shared task background 或 CLIP visual manifest。

- [Research payload](../../src/realclip_d/agent_runtime.py#L471)。
- [ExperimentContext](../../src/realclip_d/contracts.py#L36)。
- [Research 实际历史请求](../../runs/research_retry_12_20261021_090500/samples/JCP2022_065189/research/requests/001.json)：system instructions 与活动 v3 文件一致；实际 context 为 U2OS、未知剂量和处理时长。

### Observation 四阶段

`run_observation()` 转入 `run_pipeline()`；活动阶段是 baseline、guided、writer、reviewer。运行时按阶段加载 `observation_{phase}_v1.md`，直接作为 Agent instructions。

- [Observation 入口](../../src/realclip_d/agent_runtime.py#L651)。
- [阶段列表](../../src/realclip_d/observation_pipeline.py#L37)。
- [实际 prompt 加载](../../src/realclip_d/observation_pipeline.py#L269)。
- [Agent instructions](../../src/realclip_d/observation_pipeline.py#L294)。

`observation_v1.md`、`observation_v2.md`、`observation_v3.md`、`observation_v4.md` 与 `observation_output_format_v1.md` 没有进入这条四阶段运行路径。工作流对所有 prompt 文件做 hash 不代表每个文件都被加载。四阶段输出结构由各自的 strict schema 提供。

### 四份活动 prompt 的内容

- [Baseline](../../src/realclip_d/prompts/observation_baseline_v1.md#L1)：首句明确 Cell Painting。第 30–39 行明确元数据、跨通道对齐、相同 site 编号不代表同细胞、多 site 不等于独立 well。允许共同外观、control-only 与 treatment-only 发现；未声明 CLIP 终端目标。
- [Guided](../../src/realclip_d/prompts/observation_guided_v1.md#L1)：首句明确 Cell Painting，外观优先、假设仅指导。对裁图、site、跨通道和 well 级 CP 限制有明确说明；未声明 CLIP 样本单位或实际图像范围。
- [Writer](../../src/realclip_d/prompts/observation_writer_v1.md#L1)：明确写的是 “Cell Painting well comparison”。第 7 行允许以 treatment-control pattern 开头；第 16–23 行允许保留 control-only 内容。对完整观察报告而言合理；若不经后置筛选直接用作方案 A 的 caption，会出现范围不匹配。
- [Reviewer](../../src/realclip_d/prompts/observation_reviewer_v1.md#L1)：明确核验 Cell Painting treatment/control 描述，保护局部和跨通道范围，仍允许比较与 control-only 句子；未设置 treatment-only caption 要求。

## Payload 的实际可见范围

[阶段 payload 构造](../../src/realclip_d/observation_pipeline.py#L385)：

| 模块 | 实际主要输入 | 当前缺少的终端上下文 |
| --- | --- | --- |
| Research | compound、MOA、实验 context | well 级 CLIP 任务声明、CLIP 图像范围 |
| Baseline | sample_id、context、images、cp_available，附原图 | CLIP 任务声明、CLIP 图像范围、token 配置 |
| Guided | ObservationRequest 加 baseline，附图像、假设、CP catalogue | 同上 |
| Writer | 去掉 hypothesis_assessments 的 evidence | 实验 context、图像清单、CLIP 任务声明与范围 |
| Reviewer | evidence、draft、images，附原图/已有裁图 | CLIP 任务声明、CLIP 实际图像范围与 token 配置 |

`ObservationRequest` 没有 shared_task_context、目标处理 well、CLIP visual manifest 或 tokenizer budget 字段。源图像元数据已有 role、channel、field、view、well_id、stain 和裁图关系，但 `well_id` 可以为空；`SampleInput` 并没有保证恰好一个 treatment well。

因此新模块必须显式绑定目标处理 well，并校验实际 CLIP 图像清单。不能以“Observation 输入里有 treatment 图”替代目标身份和范围校验，也不能从文件名猜 well。

## 保存的真实运行核对

最后一次 Nocodazole 成功运行中的四份 `phases/*/instructions.md` 与当前四份活动 prompt 相同：

[Nocodazole 成功运行](../../runs/observation_pipeline_live_2_20261021/attempt_3_cp_query_fix_live/samples/D_R001/observation/)。

实际 request_input 没有额外的共同 developer 背景。Baseline/Guided 的 context 为 U2OS、未知剂量、未知处理时长、空 notes；没有从 notes 间接获得 CLIP 目标。

## 对新设计的影响

1. 单独定义[共享任务背景草案](shared_task_context_draft.md)，表述未来所有模块应知道的背景；明确尚未注入现有模块。
2. 保持当前 Observation 的比较报告职责。研究机制、control 观察、比较与反例仍可留在上游；终端 treatment-only 约束不禁止这些中间记录。
3. Summarize 显式收到共享任务背景、已核验最终句子、被引用观察和实际 CLIP 图像范围；不假定上游已经受过新终端目标约束。
4. 对最终句子做按命题的处理孔外观筛选与压缩。不能只接受 treatment_only 标签，也不能将相对命题机械改成绝对命题。
5. 后续若将共同背景注入各活动模块，需要另行修改 prompt 组装/输入；本轮不做这些变更。

本轮只新增设计文档。现有活动 prompt、源码和历史结果保持原样；文件校验结果记录在交付说明中。

## 本轮交付核对

- 对 `D/src/realclip_d` 下 49 份源码与现有 prompt 文件比较编辑前后 SHA-256，全部未变。
- 五份新增 Markdown 设计文档的本地链接均可解析；两份 JSON 接口示例可正常解析。
- 独立复核统一了输入字段 `target_treatment_well`、`limitations`，以及 candidate 状态、逐命题 omissions 和 provisional 输入规则。
- 未改运行代码、活动 prompt、schema 或既有结果；没有模型调用、训练或付费调用。
