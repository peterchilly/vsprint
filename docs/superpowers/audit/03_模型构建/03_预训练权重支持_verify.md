# T3.3 预训练权重支持 — 验证报告

## 验收结果：✅ 通过

- [x] 模型支持 state_dict 保存和加载
- [x] 分类头可独立于 Backbone 配置（num_speakers=None 时仅输出 embedding）
- [x] 权重初始化函数 `_init_weights()` 已实现
