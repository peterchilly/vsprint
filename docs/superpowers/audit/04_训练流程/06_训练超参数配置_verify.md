# 训练超参数配置验证测试计划

> **验证文档版本：** v1.0
> **创建日期：** 2026-06-03
> **关联任务：** 阶段四：训练流程 - 训练超参数配置

---

## 一、验证目标

### 1.1 主要目标
1. 验证 YAML 配置文件结构清晰、语义明确
2. 验证配置加载函数正确解析所有超参数
3. 验证命令行参数能正确覆盖配置文件参数
4. 验证配置验证机制能有效捕获非法参数
5. 确保配置打印输出清晰易读

### 1.2 验证范围
- YAML 配置文件格式与结构
- 配置加载函数正确性与健壮性
- 命令行参数解析与优先级
- 参数类型验证与范围检查
- 配置合并策略
- 配置打印与日志记录
- 默认值处理机制

---

## 二、测试用例

### 2.1 YAML 配置文件测试

| 用例ID | 测试项 | 输入 | 预期输出 | 优先级 |
|--------|--------|------|----------|--------|
| TC-YAML-001 | 标准配置文件加载 | 完整YAML配置文件 | 所有字段正确解析 | P0 |
| TC-YAML-002 | 最小配置文件加载 | 仅含必需字段 | 必需字段正确解析，其他使用默认值 | P0 |
| TC-YAML-003 | 配置文件不存在 | 不存在的文件路径 | 抛出 FileNotFoundError 并提示 | P0 |
| TC-YAML-004 | 配置文件格式错误 | 非法YAML语法 | 抛出解析错误并提示行号 | P0 |
| TC-YAML-005 | 空配置文件 | 空文件 | 使用全部默认值或报错 | P1 |
| TC-YAML-006 | 配置文件编码问题 | 非UTF-8编码文件 | 正确处理或报错提示 | P1 |
| TC-YAML-007 | 嵌套配置结构 | 多层嵌套配置 | 各层级正确解析 | P0 |
| TC-YAML-008 | 特殊字符处理 | 含特殊字符的值 | 正确转义和解析 | P2 |

### 2.2 配置加载函数测试

| 用例ID | 测试项 | 输入 | 预期输出 | 优先级 |
|--------|--------|------|----------|--------|
| TC-LOAD-001 | 标准加载流程 | 有效配置路径 | 返回配置字典对象 | P0 |
| TC-LOAD-002 | 类型转换正确性 | YAML中字符串数值 | 自动转换为正确类型 | P0 |
| TC-LOAD-003 | 缺失字段处理 | 缺少可选字段 | 使用预设默认值 | P0 |
| TC-LOAD-004 | 缺失必需字段 | 缺少必需字段 | 抛出异常并指明缺失字段 | P0 |
| TC-LOAD-005 | 多配置文件合并 | 主配置+覆盖配置 | 按优先级正确合并 | P1 |
| TC-LOAD-006 | 环境变量替换 | `${ENV_VAR}` 语法 | 正确替换环境变量值 | P1 |
| TC-LOAD-007 | 相对路径解析 | 相对路径配置项 | 转换为绝对路径 | P1 |
| TC-LOAD-008 | 配置缓存机制 | 重复加载同一配置 | 使用缓存或重新加载 | P2 |

### 2.3 命令行参数覆盖测试

| 用例ID | 测试项 | 输入 | 预期输出 | 优先级 |
|--------|--------|------|----------|--------|
| TC-CLI-001 | 单参数覆盖 | `--lr 0.01` | 覆盖配置文件中的lr值 | P0 |
| TC-CLI-002 | 多参数覆盖 | `--lr 0.01 --batch_size 64` | 多个值正确覆盖 | P0 |
| TC-CLI-003 | 嵌套参数覆盖 | `--optimizer.lr 0.01` | 嵌套字段正确覆盖 | P0 |
| TC-CLI-004 | 参数类型验证 | `--lr invalid` | 抛出类型错误 | P0 |
| TC-CLI-005 | 参数范围验证 | `--epochs -10` | 抛出范围错误 | P0 |
| TC-CLI-006 | 布尔参数处理 | `--use_amp true/false` | 正确解析布尔值 | P1 |
| TC-CLI-007 | 列表参数处理 | `--gpus 0,1,2,3` | 正确解析为列表 | P1 |
| TC-CLI-008 | 未知参数处理 | `--unknown_param value` | 警告或报错 | P1 |
| TC-CLI-009 | 配置文件路径指定 | `--config custom.yaml` | 加载指定配置文件 | P0 |
| TC-CLI-010 | 仅命令行模式 | 无配置文件，仅CLI参数 | 使用默认值+CLI覆盖 | P1 |

### 2.4 配置验证测试

| 用例ID | 测试项 | 输入 | 预期输出 | 优先级 |
|--------|--------|------|----------|--------|
| TC-VALID-001 | 学习率范围验证 | `lr=0.1` (正常范围) | 验证通过 | P0 |
| TC-VALID-002 | 学习率范围验证 | `lr=-0.1` 或 `lr=100` | 验证失败，提示有效范围 | P0 |
| TC-VALID-003 | batch_size范围验证 | `batch_size=0` 或负数 | 验证失败 | P0 |
| TC-VALID-004 | epochs范围验证 | `epochs=0` 或负数 | 验证失败 | P0 |
| TC-VALID-005 | 路径存在性验证 | `data_path` 不存在 | 验证失败或警告 | P1 |
| TC-VALID-006 | 参数依赖验证 | `use_amp=True` 但 `gpu_id=None` | 警告或验证失败 | P1 |
| TC-VALID-007 | 参数组合验证 | 互斥参数同时设置 | 验证失败并提示冲突 | P1 |
| TC-VALID-008 | 类型严格验证 | 字符串传入数值字段 | 验证失败 | P0 |
| TC-VALID-009 | 枚举值验证 | 优化器类型非预定义值 | 验证失败，列出有效选项 | P0 |
| TC-VALID-010 | 配置完整性验证 | 缺失关键配置项 | 列出所有缺失项 | P0 |

### 2.5 配置打印与日志测试

| 用例ID | 测试项 | 输入 | 预期输出 | 优先级 |
|--------|--------|------|----------|--------|
| TC-PRINT-001 | 标准打印格式 | 有效配置 | 格式化输出，易读 | P0 |
| TC-PRINT-002 | 结构化输出 | 嵌套配置 | 层级缩进显示 | P1 |
| TC-PRINT-003 | 敏感信息隐藏 | 含密码/token配置 | 脱敏显示 | P1 |
| TC-PRINT-004 | 配置差异对比 | 默认配置vs当前配置 | 高亮显示差异 | P2 |
| TC-PRINT-005 | JSON导出 | 配置对象 | 正确导出JSON格式 | P1 |
| TC-PRINT-006 | 日志记录 | 训练启动时 | 配置完整记录到日志 | P0 |
| TC-PRINT-007 | 配置版本记录 | 配置变更时 | 记录变更历史 | P2 |

### 2.6 默认值与回退测试

| 用例ID | 测试项 | 输入 | 预期输出 | 优先级 |
|--------|--------|------|----------|--------|
| TC-DEFAULT-001 | 学习率默认值 | 未指定lr | 使用合理默认值(如0.1) | P0 |
| TC-DEFAULT-002 | batch_size默认值 | 未指定batch_size | 使用合理默认值(如32) | P0 |
| TC-DEFAULT-003 | 优化器默认值 | 未指定optimizer | 使用默认优化器(SGD/AdamW) | P0 |
| TC-DEFAULT-004 | 调度器默认值 | 未指定scheduler | 使用默认调度器或不使用 | P1 |
| TC-DEFAULT-005 | 设备默认值 | 未指定device | 自动检测GPU或使用CPU | P1 |
| TC-DEFAULT-006 | 日志级别默认值 | 未指定log_level | 使用INFO级别 | P2 |

### 2.7 集成测试

| 用例ID | 测试项 | 输入 | 预期输出 | 优先级 |
|--------|--------|------|----------|--------|
| TC-INT-001 | 完整配置加载流程 | 配置文件+CLI参数 | 合并后配置正确 | P0 |
| TC-INT-002 | 训练循环集成 | 使用配置启动训练 | 训练正常启动，参数生效 | P0 |
| TC-INT-003 | 配置热更新 | 运行时修改配置 | 触发重新加载或拒绝 | P2 |
| TC-INT-004 | 多进程配置一致性 | 分布式训练启动 | 各进程配置一致 | P0 |
| TC-INT-005 | Checkpoint配置恢复 | 加载checkpoint | 配置正确恢复 | P1 |

---

## 三、验证方法

### 3.1 单元测试
```python
# tests/test_config.py
import pytest
import yaml
import tempfile
from pathlib import Path
from src.config import Config, load_config, validate_config

class TestYAMLLoading:
    """YAML配置加载测试"""
    
    def test_standard_config_loading(self, tmp_path):
        """TC-YAML-001: 标准配置文件加载"""
        config_content = """
        training:
          lr: 0.1
          batch_size: 32
          epochs: 100
        optimizer:
          type: SGD
          momentum: 0.9
        """
        config_file = tmp_path / "config.yaml"
        config_file.write_text(config_content)
        
        config = load_config(config_file)
        assert config.training.lr == 0.1
        assert config.training.batch_size == 32
        assert config.training.epochs == 100
        assert config.optimizer.type == "SGD"
    
    def test_config_file_not_found(self):
        """TC-YAML-003: 配置文件不存在"""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_config("nonexistent.yaml")
        assert "not found" in str(exc_info.value).lower()
    
    def test_invalid_yaml_syntax(self, tmp_path):
        """TC-YAML-004: 配置文件格式错误"""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("invalid: yaml: syntax: [")
        
        with pytest.raises(yaml.YAMLError):
            load_config(config_file)


class TestConfigLoading:
    """配置加载函数测试"""
    
    def test_type_conversion(self, tmp_path):
        """TC-LOAD-002: 类型转换正确性"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("lr: 0.1\nbatch_size: 32")
        
        config = load_config(config_file)
        assert isinstance(config.lr, float)
        assert isinstance(config.batch_size, int)
    
    def test_missing_required_field(self, tmp_path):
        """TC-LOAD-004: 缺失必需字段"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("optimizer:\n  type: SGD")
        
        with pytest.raises(ValueError) as exc_info:
            validate_config(load_config(config_file))
        assert "required" in str(exc_info.value).lower()


class TestCLIOverride:
    """命令行参数覆盖测试"""
    
    def test_single_param_override(self):
        """TC-CLI-001: 单参数覆盖"""
        base_config = {"lr": 0.1, "batch_size": 32}
        cli_args = {"lr": 0.01}
        
        merged = merge_config(base_config, cli_args)
        assert merged["lr"] == 0.01
        assert merged["batch_size"] == 32
    
    def test_nested_param_override(self):
        """TC-CLI-003: 嵌套参数覆盖"""
        base_config = {"optimizer": {"lr": 0.1, "type": "SGD"}}
        cli_args = {"optimizer.lr": 0.01}
        
        merged = merge_config(base_config, cli_args)
        assert merged["optimizer"]["lr"] == 0.01
        assert merged["optimizer"]["type"] == "SGD"
    
    def test_invalid_param_type(self):
        """TC-CLI-004: 参数类型验证"""
        with pytest.raises(TypeError):
            parse_cli_arg("--lr", "invalid")


class TestConfigValidation:
    """配置验证测试"""
    
    def test_lr_range_validation(self):
        """TC-VALID-001/002: 学习率范围验证"""
        assert validate_lr(0.1) is True
        with pytest.raises(ValueError):
            validate_lr(-0.1)
        with pytest.raises(ValueError):
            validate_lr(100)
    
    def test_batch_size_validation(self):
        """TC-VALID-003: batch_size范围验证"""
        with pytest.raises(ValueError):
            validate_batch_size(0)
        with pytest.raises(ValueError):
            validate_batch_size(-16)
    
    def test_optimizer_type_validation(self):
        """TC-VALID-009: 枚举值验证"""
        valid_types = ["SGD", "Adam", "AdamW"]
        with pytest.raises(ValueError) as exc_info:
            validate_optimizer_type("InvalidOptimizer", valid_types)
        assert "SGD" in str(exc_info.value)


class TestConfigPrinting:
    """配置打印测试"""
    
    def test_formatted_output(self):
        """TC-PRINT-001: 标准打印格式"""
        config = Config(lr=0.1, batch_size=32)
        output = format_config(config)
        
        assert "lr" in output
        assert "0.1" in output
        assert "batch_size" in output
    
    def test_sensitive_info_masking(self):
        """TC-PRINT-003: 敏感信息隐藏"""
        config = Config(api_key="secret_key_123", lr=0.1)
        output = format_config(config, mask_sensitive=True)
        
        assert "secret_key_123" not in output
        assert "***" in output
```

### 3.2 集成测试
- **测试环境：** Python 3.8+, PyTorch 2.0+
- **测试数据：** 标准配置模板、边界值配置
- **验证工具：** pytest, pytest-cov

### 3.3 配置验证脚本
```python
# scripts/validate_config.py
"""配置验证脚本"""

import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, List

class ConfigValidator:
    """配置验证器"""
    
    # 参数定义与约束
    PARAM_SCHEMA = {
        "training.lr": {
            "type": float,
            "min": 1e-7,
            "max": 10.0,
            "default": 0.1,
            "description": "学习率"
        },
        "training.batch_size": {
            "type": int,
            "min": 1,
            "max": 1024,
            "default": 32,
            "description": "批次大小"
        },
        "training.epochs": {
            "type": int,
            "min": 1,
            "max": 10000,
            "default": 100,
            "description": "训练轮数"
        },
        "optimizer.type": {
            "type": str,
            "choices": ["SGD", "Adam", "AdamW", "RMSprop"],
            "default": "SGD",
            "description": "优化器类型"
        },
        "optimizer.momentum": {
            "type": float,
            "min": 0.0,
            "max": 1.0,
            "default": 0.9,
            "description": "动量系数"
        },
        "optimizer.weight_decay": {
            "type": float,
            "min": 0.0,
            "max": 1.0,
            "default": 1e-4,
            "description": "权重衰减"
        },
        "scheduler.type": {
            "type": str,
            "choices": ["CosineAnnealingLR", "StepLR", "ExponentialLR", "None"],
            "default": "CosineAnnealingLR",
            "description": "学习率调度器"
        },
        "warmup.epochs": {
            "type": int,
            "min": 0,
            "max": 50,
            "default": 5,
            "description": "预热轮数"
        },
    }
    
    def validate(self, config: Dict[str, Any]) -> List[str]:
        """验证配置，返回错误列表"""
        errors = []
        
        for param_path, schema in self.PARAM_SCHEMA.items():
            value = self._get_nested(config, param_path)
            
            if value is None:
                if "default" not in schema:
                    errors.append(f"缺少必需参数: {param_path}")
                continue
            
            # 类型检查
            if not isinstance(value, schema["type"]):
                errors.append(f"{param_path}: 类型错误，期望 {schema['type'].__name__}")
                continue
            
            # 范围检查
            if "min" in schema and value < schema["min"]:
                errors.append(f"{param_path}: 值 {value} 小于最小值 {schema['min']}")
            if "max" in schema and value > schema["max"]:
                errors.append(f"{param_path}: 值 {value} 大于最大值 {schema['max']}")
            
            # 枚举检查
            if "choices" in schema and value not in schema["choices"]:
                errors.append(f"{param_path}: 无效值 '{value}'，有效选项: {schema['choices']}")
        
        return errors
    
    def _get_nested(self, config: Dict, path: str) -> Any:
        """获取嵌套配置值"""
        keys = path.split(".")
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value

def main():
    parser = argparse.ArgumentParser(description="配置验证工具")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    validator = ConfigValidator()
    errors = validator.validate(config)
    
    if errors:
        print("❌ 配置验证失败:")
        for error in errors:
            print(f"  - {error}")
        return 1
    else:
        print("✅ 配置验证通过")
        return 0

if __name__ == "__main__":
    exit(main())
```

---

## 四、验收标准（Pass Criteria）

### 4.1 必须满足（P0级别全部通过）
- [ ] YAML配置文件正确加载，所有字段正确解析
- [ ] 配置文件不存在时抛出明确错误
- [ ] YAML格式错误时抛出解析错误并提示位置
- [ ] 命令行参数正确覆盖配置文件参数
- [ ] 嵌套参数（如 `optimizer.lr`）正确覆盖
- [ ] 参数类型验证正常工作
- [ ] 参数范围验证正常工作
- [ ] 配置打印输出清晰可读
- [ ] 训练流程正确使用加载的配置

### 4.2 应该满足（P1级别80%以上通过）
- [ ] 多配置文件合并正确
- [ ] 环境变量替换功能正常
- [ ] 敏感信息脱敏显示
- [ ] 布尔/列表参数正确解析
- [ ] 参数组合验证正常
- [ ] checkpoint配置恢复正确

### 4.3 可选满足（P2级别）
- [ ] 配置差异对比功能
- [ ] 配置变更历史记录
- [ ] 配置热更新支持
- [ ] 特殊字符处理

### 4.4 量化指标

| 指标 | 目标值 | 测量方法 |
|------|--------|----------|
| 单元测试覆盖率 | ≥ 90% | pytest-cov |
| 集成测试通过率 | 100% (P0) | pytest report |
| 配置加载时间 | < 100ms | time测量 |
| 参数验证准确率 | 100% | 边界值测试 |
| 错误提示可理解性 | 人工评审通过 | 评审会议 |

---

## 五、自动化建议

### 5.1 CI/CD 集成
```yaml
# .github/workflows/test_config.yml
name: Config Module Tests

on:
  push:
    paths:
      - 'src/config/**'
      - 'configs/**'
      - 'tests/test_config.py'
  pull_request:
    paths:
      - 'src/config/**'
      - 'configs/**'

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.8', '3.9', '3.10', '3.11']
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov pyyaml
      
      - name: Run config unit tests
        run: |
          pytest tests/test_config.py -v --cov=src/config --cov-report=xml
      
      - name: Validate all config files
        run: |
          python scripts/validate_config.py --config configs/default.yaml
          python scripts/validate_config.py --config configs/production.yaml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

### 5.2 自动化测试脚本
```bash
#!/bin/bash
# scripts/test_config.sh

echo "=== 配置模块验证测试 ==="

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# 1. 运行单元测试
echo -e "\n[1/5] 运行单元测试..."
pytest tests/test_config.py -v
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 单元测试通过${NC}"
else
    echo -e "${RED}✗ 单元测试失败${NC}"
    exit 1
fi

# 2. 验证所有配置文件
echo -e "\n[2/5] 验证所有配置文件..."
for config in configs/*.yaml; do
    python scripts/validate_config.py --config "$config"
    if [ $? -ne 0 ]; then
        echo -e "${RED}✗ 配置文件验证失败: $config${NC}"
        exit 1
    fi
done
echo -e "${GREEN}✓ 所有配置文件验证通过${NC}"

# 3. 测试配置加载性能
echo -e "\n[3/5] 测试配置加载性能..."
python -c "
import time
from src.config import load_config

start = time.time()
for _ in range(100):
    load_config('configs/default.yaml')
elapsed = time.time() - start

avg_time = elapsed / 100 * 1000
print(f'平均加载时间: {avg_time:.2f}ms')
assert avg_time < 100, f'加载时间过长: {avg_time:.2f}ms'
"

# 4. 测试命令行参数覆盖
echo -e "\n[4/5] 测试命令行参数覆盖..."
python -c "
import sys
sys.argv = ['train.py', '--config', 'configs/default.yaml', '--lr', '0.01']
from src.config import parse_args, load_config

args = parse_args()
config = load_config(args.config)
config = merge_cli_args(config, args)
assert config.training.lr == 0.01, 'CLI参数覆盖失败'
print('✓ CLI参数覆盖正确')
"

# 5. 生成测试报告
echo -e "\n[5/5] 生成测试报告..."
pytest tests/test_config.py --html=reports/config_test.html --self-contained-html

echo -e "\n${GREEN}=== 所有测试完成 ===${NC}"
```

### 5.3 配置文件模板生成器
```python
# scripts/generate_config_template.py
"""生成配置文件模板"""

import yaml
from typing import Dict, Any

def generate_config_template() -> Dict[str, Any]:
    """生成完整配置模板"""
    return {
        "training": {
            "lr": 0.1,
            "batch_size": 32,
            "epochs": 100,
            "num_workers": 4,
            "seed": 42,
        },
        "optimizer": {
            "type": "SGD",
            "momentum": 0.9,
            "weight_decay": 1e-4,
            "nesterov": True,
        },
        "scheduler": {
            "type": "CosineAnnealingLR",
            "T_max": 100,
            "eta_min": 1e-6,
        },
        "warmup": {
            "epochs": 5,
            "start_lr": 1e-7,
            "mode": "linear",
        },
        "model": {
            "name": "ERes2NetV2",
            "num_classes": 1000,
            "pretrained": True,
        },
        "data": {
            "train_path": "/path/to/train",
            "val_path": "/path/to/val",
            "input_size": 224,
        },
        "logging": {
            "log_dir": "./logs",
            "save_dir": "./checkpoints",
            "log_level": "INFO",
        },
        "device": {
            "gpu_id": 0,
            "use_amp": True,
        },
    }

def save_template(path: str):
    """保存模板到文件"""
    template = generate_config_template()
    with open(path, "w") as f:
        yaml.dump(template, f, default_flow_style=False, sort_keys=False)
    print(f"配置模板已保存到: {path}")

if __name__ == "__main__":
    save_template("configs/template.yaml")
```

---

## 六、风险点与缓解措施

### 6.1 技术风险

| 风险ID | 风险描述 | 可能影响 | 概率 | 缓解措施 |
|--------|----------|----------|------|----------|
| R-001 | YAML解析器版本差异 | 配置加载失败 | 低 | 明确PyYAML版本要求≥5.1 |
| R-002 | 参数类型隐式转换 | 运行时错误 | 中 | 严格类型验证，拒绝隐式转换 |
| R-003 | 嵌套配置覆盖错误 | 参数值错误 | 中 | 完整的单元测试覆盖 |
| R-004 | 环境变量未设置 | 配置解析失败 | 低 | 提供默认值，明确文档说明 |
| R-005 | 配置版本不兼容 | 训练失败 | 中 | 添加配置版本号，兼容性检查 |

### 6.2 使用风险

| 风险ID | 风险描述 | 可能影响 | 概率 | 缓解措施 |
|--------|----------|----------|------|----------|
| R-006 | 参数拼写错误 | 使用错误值 | 高 | 提供参数自动补全和拼写检查 |
| R-007 | 单位混淆（如lr vs weight_decay） | 训练效果差 | 中 | 添加单位说明和范围警告 |
| R-008 | 配置文件路径错误 | 找不到配置 | 中 | 支持相对路径，提供友好错误提示 |
| R-009 | 敏感信息泄露 | 安全风险 | 中 | 自动检测并脱敏敏感字段 |
| R-010 | 配置冗余/冲突 | 参数设置混乱 | 低 | 提供配置检查和合并建议 |

### 6.3 维护风险

| 风险ID | 风险描述 | 可能影响 | 概率 | 缓解措施 |
|--------|----------|----------|------|----------|
| R-011 | 新参数未更新验证规则 | 无效参数被接受 | 中 | 参数与验证规则同步更新 |
| R-012 | 文档与代码不一致 | 使用困惑 | 高 | 文档从代码自动生成 |
| R-013 | 默认值不合理 | 训练效果差 | 中 | 提供多套预设配置 |

---

## 七、验证执行清单

### 7.1 执行前准备
- [ ] 确认 Python 版本 ≥ 3.8
- [ ] 确认 PyYAML 版本 ≥ 5.1
- [ ] 准备测试配置文件（正常、边界、错误）
- [ ] 安装测试依赖（pytest, pytest-cov）
- [ ] 配置测试环境

### 7.2 执行顺序
1. **单元测试** → 验证各组件独立功能
2. **集成测试** → 验证组件协同工作
3. **配置验证脚本** → 验证所有配置文件
4. **性能测试** → 验证加载性能
5. **文档检查** → 验证配置文档准确性

### 7.3 验收确认
- [ ] 所有P0测试用例通过
- [ ] P1测试通过率 ≥ 80%
- [ ] 配置加载时间 < 100ms
- [ ] 代码覆盖率 ≥ 90%
- [ ] 所有配置文件验证通过
- [ ] 文档审查完成
- [ ] Code Review 完成

---

## 八、附录

### 8.1 配置文件完整示例

```yaml
# configs/default.yaml
# 训练超参数配置模板

# 训练基础参数
training:
  lr: 0.1                    # 学习率
  batch_size: 32             # 批次大小
  epochs: 100                # 训练轮数
  num_workers: 4             # 数据加载进程数
  seed: 42                   # 随机种子

# 优化器配置
optimizer:
  type: SGD                  # 优化器类型: SGD, Adam, AdamW
  momentum: 0.9              # 动量系数
  weight_decay: 1.0e-4       # 权重衰减
  nesterov: true             # 是否使用Nesterov动量

# 学习率调度器
scheduler:
  type: CosineAnnealingLR    # 调度器类型
  T_max: 100                 # 周期长度
  eta_min: 1.0e-6            # 最小学习率

# 预热配置
warmup:
  epochs: 5                  # 预热轮数
  start_lr: 1.0e-7           # 起始学习率
  mode: linear               # 预热模式: linear, cosine

# 模型配置
model:
  name: ERes2NetV2           # 模型名称
  num_classes: 1000          # 分类数
  pretrained: true           # 是否使用预训练权重
  checkpoint: null           # 恢复训练的checkpoint路径

# 数据配置
data:
  train_path: ./data/train   # 训练数据路径
  val_path: ./data/val       # 验证数据路径
  input_size: 224            # 输入尺寸
  augment: true              # 是否使用数据增强

# 日志配置
logging:
  log_dir: ./logs            # 日志目录
  save_dir: ./checkpoints    # 模型保存目录
  log_level: INFO            # 日志级别
  save_interval: 10          # 保存间隔（epoch）
  tensorboard: true          # 是否使用TensorBoard

# 设备配置
device:
  gpu_id: 0                  # GPU ID，-1表示使用CPU
  use_amp: true              # 是否使用混合精度
  deterministic: false       # 是否使用确定性算法
```

### 8.2 命令行参数参考

```bash
# 完整训练命令示例
python train.py \
    --config configs/default.yaml \
    --training.lr 0.01 \
    --training.batch_size 64 \
    --training.epochs 50 \
    --optimizer.type AdamW \
    --optimizer.weight_decay 0.01 \
    --scheduler.type StepLR \
    --warmup.epochs 3 \
    --device.gpu_id 0,1,2,3 \
    --device.use_amp true

# 仅使用命令行参数（无配置文件）
python train.py \
    --no-config \
    --training.lr 0.1 \
    --training.batch_size 32
```

### 8.3 配置验证规则表

| 参数路径 | 类型 | 范围/选项 | 默认值 | 必需 |
|----------|------|-----------|--------|------|
| training.lr | float | [1e-7, 10.0] | 0.1 | 是 |
| training.batch_size | int | [1, 1024] | 32 | 是 |
| training.epochs | int | [1, 10000] | 100 | 是 |
| training.seed | int | [0, 2^31-1] | 42 | 否 |
| optimizer.type | str | {SGD, Adam, AdamW, RMSprop} | SGD | 是 |
| optimizer.momentum | float | [0.0, 1.0] | 0.9 | 否 |
| optimizer.weight_decay | float | [0.0, 1.0] | 1e-4 | 否 |
| scheduler.type | str | {CosineAnnealingLR, StepLR, ExponentialLR, None} | CosineAnnealingLR | 否 |
| warmup.epochs | int | [0, 50] | 5 | 否 |
| device.gpu_id | int或list | [-1, 可用GPU数] | 0 | 否 |
| device.use_amp | bool | - | true | 否 |

### 8.4 错误信息参考

| 错误码 | 错误信息 | 原因 | 解决方案 |
|--------|----------|------|----------|
| E001 | Config file not found: {path} | 配置文件不存在 | 检查文件路径 |
| E002 | Invalid YAML syntax at line {n} | YAML格式错误 | 修正YAML语法 |
| E003 | Missing required field: {field} | 缺少必需字段 | 添加必需字段 |
| E004 | Invalid value for {field}: {value} | 参数值无效 | 使用有效值 |
| E005 | Type mismatch for {field}: expected {type} | 类型不匹配 | 使用正确类型 |
| E006 | Value out of range for {field}: [{min}, {max}] | 值超出范围 | 使用范围内的值 |
| E007 | Invalid choice for {field}: {value}. Valid options: {options} | 无效选项 | 使用有效选项 |
| E008 | CLI argument parse error: {arg} | CLI参数解析失败 | 检查参数格式 |

### 8.5 相关文档
- [PyYAML官方文档](https://pyyaml.org/wiki/PyYAMLDocumentation)
- [Python argparse文档](https://docs.python.org/3/library/argparse.html)
- [Hydra配置框架](https://hydra.cc/)
- [OmegaConf文档](https://omegaconf.readthedocs.io/)

---

**文档维护者：** 训练流程组
**最后更新：** 2026-06-03