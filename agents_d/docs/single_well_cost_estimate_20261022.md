# 每个处理 well 的费用估算（2026-10-22）

单位是一个处理 well 的全部已提供 sites/channels、对应 control 的 Observation，再生成一份该处理 well 的短 description。Research 可以按化合物和实验条件复用。这里按两例实际完成调用的 token 用量回算，不包含 CLIP 训练、存储或图像下载费用。

当前 gpt-6-luna 标准短上下文费率（每百万 token）：普通输入 $0.10、缓存读取 $0.01、默认 30 分钟缓存写入 $0.125、输出 $0.50。推理 token 已包含在输出用量中。来源：[OpenAI 定价](https://developers.openai.com/api/docs/pricing)、[缓存计费与默认 TTL](https://developers.openai.com/api/docs/guides/prompt-caching)。

| 生成范围 | Nocodazole 标准用量估算 | Chloroquine 标准用量估算 |
| --- | ---: | ---: |
| 只跑一次 Summarize | $0.007458 | $0.006838 |
| 完整四阶段 Observation | $0.084268 | $0.089485 |
| 只跑 Research | $0.017766 | $0.015155 |
| 已有 Research：Observation + Summarize | $0.091725 | $0.096323 |
| 从头 Research + Observation + Summarize | $0.109491 | $0.111478 |

预算账本仍按输入 $0.25/M、输出 $0.75/M 预留和扣减，不单独折算缓存，属于较保守的运行预算估算。它与上表标准费率回算不同，两者都没有与供应商账单对账。

| 范围 | Nocodazole 保守预算估算 | Chloroquine 保守预算估算 |
| --- | ---: | ---: |
| Observation + Summarize | $0.263766 | $0.285121 |
| 全部从头 | $0.419022 | $0.428866 |

目前 Research 已有，后续可按每 well 标准用量约 $0.09–$0.10 理解成本，按约 $0.30/well 留运行预算。只压缩已有 Observation 的一次调用约 $0.007。

样本为 Nocodazole 6 sites 和 Chloroquine 9 sites。Observation 各 6 次模型调用；Research 各 14 次。Summarize 估算各取一次生成，排除旧版引用修正。新的纯文本版本已取消引用输出和自动修正，但尚未付费测量；其费用只能暂用这些既有样本作参考，实际随 site/channel 数、推理长度及缓存命中变化。

本次只修改 Summarize 并离线检查，没有运行批量生成、付费调用或 CLIP 训练。原始调用路径、缓存用量和计算结果见 [cost_estimate.json](<PROJECT_ROOT>/D/runs/summarize_plain_output_offline_20261022/cost_estimate.json)。
