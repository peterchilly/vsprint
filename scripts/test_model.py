"""
ERes2NetV2 模型验证脚本

验证：
1. 前向传播
2. 输出维度
3. 参数量
4. 梯度反传
5. AAM-Softmax 损失
"""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.eres2netv2 import (
    create_eres2netv2,
    eres2netv2_34,
    eres2netv2_50,
    eres2netv2_101,
    eres2netv2_152,
)
from models.losses import AAMSoftmaxLoss, ArcFaceLoss


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def test_forward():
    print("=" * 60)
    print("🧪 测试 1: 前向传播")
    print("=" * 60)

    variants = ["34", "50", "101", "152"]

    for v in variants:
        model = create_eres2netv2(variant=v, n_mels=80, embedding_dim=192)
        x = torch.randn(2, 1, 80, 300)  # (batch, channel, n_mels, time)

        embedding, logits = model(x)

        assert embedding.shape == (2, 192), f"ERes2NetV2-{v}: embedding shape mismatch: {embedding.shape}"
        assert logits is None, f"ERes2NetV2-{v}: logits should be None"

        params = count_params(model)
        print(f"  ✅ ERes2NetV2-{v}: input(2,1,80,300) → embedding(2,192), 参数: {params/1e6:.2f}M")


def test_with_classifier():
    print("\n" + "=" * 60)
    print("🧪 测试 2: 带分类头")
    print("=" * 60)

    model = create_eres2netv2(variant="34", n_mels=80, embedding_dim=192, num_speakers=100)
    x = torch.randn(2, 1, 80, 300)

    embedding, logits = model(x)

    assert embedding.shape == (2, 192)
    assert logits.shape == (2, 100), f"Logits shape mismatch: {logits.shape}"
    print(f"  ✅ ERes2NetV2-34 + classifier: input(2,1,80,300) → embedding(2,192) + logits(2,100)")


def test_backward():
    print("\n" + "=" * 60)
    print("🧪 测试 3: 梯度反传")
    print("=" * 60)

    model = create_eres2netv2(variant="34", n_mels=80, embedding_dim=192, num_speakers=50)
    x = torch.randn(2, 1, 80, 300, requires_grad=True)
    labels = torch.tensor([0, 1])

    embedding, logits = model(x)
    loss_fn = AAMSoftmaxLoss(embedding_dim=192, num_speakers=50, margin=0.2, scale=30.0)
    loss = loss_fn(embedding, labels)

    loss.backward()

    # 检查梯度
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"

    print(f"  ✅ 梯度反传正常，loss={loss.item():.4f}")
    print(f"  ✅ 无 NaN/Inf 梯度")


def test_aam_softmax():
    print("\n" + "=" * 60)
    print("🧪 测试 4: AAM-Softmax 损失")
    print("=" * 60)

    loss_fn = AAMSoftmaxLoss(embedding_dim=192, num_speakers=100, margin=0.2, scale=30.0)
    embedding = torch.randn(32, 192)
    labels = torch.randint(0, 100, (32,))

    loss = loss_fn(embedding, labels)
    assert loss.dim() == 0
    print(f"  ✅ AAM-Softmax: batch=32, loss={loss.item():.4f}")

    # 对比标准 CE
    ce = torch.nn.CrossEntropyLoss()
    logits = torch.randn(32, 100)
    ce_loss = ce(logits, labels)
    print(f"  ✅ CrossEntropy 对比: loss={ce_loss.item():.4f}")


def test_model_variants():
    print("\n" + "=" * 60)
    print("🧪 测试 5: 各变体参数量对比")
    print("=" * 60)

    for variant, factory in [("34", eres2netv2_34), ("50", eres2netv2_50),
                              ("101", eres2netv2_101), ("152", eres2netv2_152)]:
        model = factory(n_mels=80, embedding_dim=192)
        params = count_params(model)
        print(f"  ERes2NetV2-{variant:>3}: {params/1e6:.2f}M 参数")


def test_gpu():
    print("\n" + "=" * 60)
    print("🧪 测试 6: GPU 推理")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  ⏭️  GPU 不可用，跳过")
        return

    model = eres2netv2_34(n_mels=80, embedding_dim=192, num_speakers=50).cuda()
    x = torch.randn(8, 1, 80, 300).cuda()
    labels = torch.randint(0, 50, (8,)).cuda()

    embedding, logits = model(x)
    loss_fn = AAMSoftmaxLoss(embedding_dim=192, num_speakers=50).cuda()
    loss = loss_fn(embedding, labels)
    loss.backward()

    print(f"  ✅ GPU 推理正常，loss={loss.item():.4f}")
    print(f"  ✅ GPU 显存占用: {torch.cuda.memory_allocated()/1024**2:.1f} MB")


if __name__ == "__main__":
    test_forward()
    test_with_classifier()
    test_backward()
    test_aam_softmax()
    test_model_variants()
    test_gpu()

    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)
