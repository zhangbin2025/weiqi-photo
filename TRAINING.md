# 训练重放（Training Replay）

本仓库不含训练数据，但**随仓库签入已训练好的 CNN 模型**
`training/out-cnn/cnn.onnx`（demo 直接加载它）。该模型由
`training/cnn/train_cnn.py` 在同一份外部标注数据集上可回放训练产出；本文说明
数据格式、依赖、命令和产出，照做即可重放训练、复现签入的模型。

推理侧见根目录 `README.md`。

---

## 1. 数据目录

训练入口通过 `--data <目录>` 接收一个**外部**数据目录（不在本仓库内）。
目录里放一个 `dataset.json`，以及它引用的图片文件和真值文件：

```
<数据目录>/
  dataset.json
  images/      *.png / *.jpg   （棋盘照片）
  truth/       *.txt           （逐交叉点真值）
```

`dataset.json` 结构：

```json
{
  "samples": [
    {
      "sample_id": "0001",
      "split": "train",
      "image": "images/0001.png",
      "truth": "truth/0001.txt",
      "quad": [[20,20],[170,30],[180,175],[25,165]]
    },
    { "sample_id": "0002", "split": "val", ... }
  ]
}
```

字段约定：

| 字段 | 含义 |
|---|---|
| `sample_id` | 唯一标识，仅用于日志/追溯 |
| `split` | `train` / `val` / `test`；**`test` 会被 `load_corpus()` 直接过滤，永不参与训练** |
| `image` | 图片相对路径（相对数据目录）；RGB，任意尺寸，加载后 `convert("RGB")` 归一化到 `[0,1]` |
| `truth` | 真值文本相对路径，格式见 §2 |
| `quad` | 四个**可下子区域**角点 `[x,y]`，顺序 **左上 → 右上 → 右下 → 左下**（像素坐标，原图尺度）。透视校正按此四边形把棋盘拉正 |

- 所有 `image` / `truth` 路径都相对数据目录解析成绝对路径，数据目录可放在任意位置。
- **同一次训练里所有图片必须是同一路数**（脚本用第一个样本的真值行数推断路数 `lines`，
  并按 `lines*lines` 预分配 patch 数组）。19/13/9 路请分开跑。

数据来源需自行准备与标注（照片 + 四角 + 逐点真值）。本开源仓库不分发任何数据。

---

## 2. 真值格式

`truth` 是纯文本文件：**N 行、每行 N 个字符**（N = 路数），第 r 行第 c 列对应
棋盘第 r 行第 c 列交叉点：

| 字符 | 类别 | 索引 |
|---|---|---|
| `.` | 空 | 0 |
| `B` | 黑子 | 1 |
| `W` | 白子 | 2 |

例（9 路，前 3 行示意）：

```
.BW.BW.BW
.........
BB.WW.B..
...（共 9 行）
```

`truth_matrix()` 直接按字符映射成 `int8` 矩阵，不做合法性补全；请确保每行等长、
行列数一致、只含 `.` `B` `W`。

---

## 3. 依赖（训练时）

推理侧 `inference/` 是零第三方依赖的原生 ES modules；**只有训练需要装 Python 包**。
以下版本为实际可复现组合（编译机 venv，Python 3.12.3）：

| 包 | 版本 | 用途 |
|---|---|---|
| torch | 2.15.0+cpu | 模型定义 / 训练 / `torch.onnx.export` |
| onnx | 1.23.2 | ONNX 导出（torch.onnx 使用） |
| numpy | 2.5.4 | 数组 / 几何 |
| Pillow | 13.0.0 | 图片解码 |

依赖已完整列入仓库根 `requirements.txt`（含 torch CPU 轮的 `--extra-index-url`），
一条命令即可装齐：

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

若手工安装：

```bash
pip install numpy==2.5.4 Pillow==13.0.0 onnx==1.23.2
pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu
```

各包许可以其自带 `dist-info/licenses/` 为准。

无 GPU 亦可（纯 CPU，2 核 / 7.5G 内存约数分钟，视样本量而定）。

---

## 4. 训练命令

在**仓库根目录**执行（`-m training.cnn.train_cnn` 依赖包相对导入）：

```bash
python3 -m training.cnn.train_cnn \
  --data /path/to/你的数据目录 \
  --out training/out-cnn \
  --epochs 5 --batch 512 --lr 1e-3 --cell 40
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--data` | **必填** | §1 的数据目录 |
| `--out` | `training/out-cnn` | 产出目录 |
| `--epochs` | 5 | 训练轮数 |
| `--batch` | 512 | batch size |
| `--lr` | 1e-3 | Adam 学习率 |
| `--cell` | 40 | warp 后每格像素数 |

**`--cell` 必须与推理侧一致**：`inference/src/cnn_inference.mjs` 的
`CELL_DEFAULT` 和 `PATCH_SIZE` 均为 40，`train_cnn.py` 的 `PATCH` 也是 40。
改动需三处同步，否则切出的 patch 与模型不匹配。

流程：`load_corpus` 读样本 → `decode_image` 解码 → `truth_matrix` 读真值 →
`warp_perspective` 透视拉正（产出 `V` 灰度）→ 每个交叉点切 40×40 灰度 patch →
LeNet 风格 CNN（3 类：`./B/W`）→ 导出 ONNX。

---

## 5. 产出

全部写到 `--out` 目录。其中 **`cnn.onnx` 已随仓库签入**（demo 直接加载），
其余为中间产物，被 `.gitignore` 屏蔽，不签入：

| 文件 | 说明 |
|---|---|
| `cnn.onnx` | **已签入**的推理模型（约 533 KiB）；demo 直接加载 |
| `cnn.pt` | PyTorch `state_dict` 存档（浏览器用不到），中间产物，不签入 |
| `cnn_meta.json` | 架构 / 超参 / `val_accuracy` 等元信息，不签入 |
| `cnn_eval.json` | 验证集混淆矩阵、per-class precision/recall，不签入 |

日志会打印每轮 `loss / train_acc / val_acc` 及最终混淆矩阵。
用相同数据集与超参重放训练，`cnn.onnx` 应与签入版本一致；重跑会覆盖该文件，
可用 `git diff -- training/out-cnn/cnn.onnx` 核对。

---

## 6. 与 demo 联调

`inference/tools/serve.mjs` 的静态根是**仓库根目录**，默认把
`/training/out-cnn/cnn.onnx` 作为模型 URL。模型已随仓库签入，无需先训练、
也无需移动文件即可直接跑 demo：

```bash
cd inference && node tools/serve.mjs
# 浏览器打开 http://127.0.0.1:8899/inference/app/
# 导入照片 → 点 4 角 → 点「识别 (CNN)」
```

ONNX Runtime Web 运行时从 CDN（jsDelivr，onnxruntime-web@1.19.2）加载，
首次需联网；模型文件本身已随仓库签入、随仓库分发。
若 `cnn.onnx` 被误删，demo 会报 `ONNX 加载失败: 404`。
