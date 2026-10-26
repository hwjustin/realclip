# Summarize 模块设计稿

状态：仅设计，尚未接入运行。检查日期：2026-10-22。

本轮只审查当前实际加载的 prompt，并新增这组设计文档。现有 Research、Observation 四阶段、运行 prompt、输出合同和历史结果均不改写；没有模型生成、训练或付费调用。

本次审查结论：Research 与四个 Observation 阶段都明确提到 Cell Painting；它们目前没有明确收到“一个处理 well 的多 site、多 channel 图像集合 ↔ 一份处理孔外观短 caption”的 CLIP 任务目标，也没有共享的 CLIP 输入清单。Writer 当前明确负责完整的 well comparison，这符合它现有的中间报告职责。

- [实际 prompt 与 payload 审查](prompt_context_audit.md)：现状、加载路径、缺口和证据。
- [共享任务背景草案](shared_task_context_draft.md)：未来各阶段应知道的共同背景，尚未注入活动模块。
- [Summarize 流程与接口](flow_design.md)：后置模块输入、证据筛选、长度、输出与检查。
- [Summarize 完整 prompt 草案](summarize_prompt_draft.md)：英文设计稿，尚未加载为 Agent instructions。

拟议配对单位是一个处理 well。control 和 CP 可帮助形成上游观察与限定，最终训练 caption 只保留实际处理图输入所支持的外观。中间报告仍可保存比较、control-only 记录和测量限制。

共享任务背景与实际可访问证据是两层约定。让所有模块知道最终用途，不要求 Baseline 看到假设或 CP 数值，也不要求 Writer 重新看图。后续若统一注入共享背景，需要单独实现；本轮的设计稿不代表已有模块已经获得该背景。
