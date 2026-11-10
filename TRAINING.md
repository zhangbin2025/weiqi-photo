# 训练重放（Training Replay）

本仓库**不含训练数据**，但公开完整的训练过程：数据来源、划分、代码、依赖版本、
命令与校验方法都在这里。随仓库签入的 `training/out-cnn/cnn.onnx` 就是用下面这条
命令、用同一份 gomrade 数据集训练产出的；在同样的环境下重跑，产出的模型文件
**逐字节一致**。

推理侧见根目录 `README.md`。

---

## 1. 数据来源与致谢

训练数据全部来自 **gomrade**：

> **gomrade dataset — Go / Baduk images with labels**
> 作者：davids1992（Kaggle）
> <https://www.kaggle.com/datasets/davids1992/gomrade-dataset-go-baduk-images-with-labels>
> 许可：[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)

**特别感谢 gomrade 数据集的作者与所有贡献者。** 没有这份公开标注数据，就没有
本项目里的 CNN 模型：它提供了棋盘照片、逐交叉点真值和四角标定，使得"拍照还原
棋局"可以在完全开放、可回放的条件下训练出来。

按 CC BY-NC-ND 4.0：

- **BY（署名）**：这里明确标注了作者、来源链接与许可证；
- **NC（非商业）**：gomrade 数据仅可用于非商业目的。本仓库的 MIT 许可**只覆盖
  代码**，不覆盖该数据集，也不授予任何商业使用权；
- **ND（禁止演绎）**：本仓库**不分发、不重编码、不裁剪保存**任何 gomrade 文件。
  训练时只做内存态技术转换（把空格分隔的真值文本解析成矩阵、从 YAML 读出四个
  角点坐标），不产生任何数据集衍生文件；
- 由该数据集训练得到的模型权重，其权利归属与可用范围受数据集许可约束，请以
  原作者与 Kaggle 页面的条款为准。请自行核对许可证后再使用。

**本仓库只分发代码与模型，不分发数据。** 数据请从上面的 Kaggle 链接自行获取。

### 实际训练输入（把来源说透）

签入模型实际使用的数据是 gomrade 的一个子集，由 `training/cnn/split.tsv`
（seed `20261105` 的按 unit 划分）确定：

| 项 | 值 |
|---|---|
| 参与训练/验证的帧 | **1620**（`dataset/` 951 帧 + `dataset2/` 669 帧）|
| train / val | **1325 / 295** 帧 |
| 涉及的 unit（一局棋一个目录）| **39**（22 train / 17 val）|
| 每帧切出的 patch | 19 × 19 = 361，共 **584820** 个 40×40 单通道 patch |
| 未参与 | `split.tsv` 标注为封存 test 的 6 个 gomrade unit / 506 帧、无四角或无疑似重复的子目录 |
| 图片 EXIF | 这 39 个 unit 全部为无 tag 或 tag=1，**不做任何朝向旋转** |

`training/cnn/split.tsv` 共 61 行，覆盖 gomrade 解压后的全部 61 个子目录：
22 个 `train`、17 个 `val`、6 个 `test`（封存，从不加载）、16 个 `exclude`。
16 个 `exclude` 里，`g1:t2` / `g1:t3` 虽有图片+真值配对但被划分排除，12 个
`g2:13`…`g2:25` 子目录只有 yml 四角、没有图片+真值配对，另有
`synthesized_moves` / `g2:27` 连 yml 也没有。

gomrade 原始目录里还有不参与训练的文件：438 个 `.old` 快照、`colors.txt` /
`pts.txt`（棋盘点位/配色记录，非真值矩阵）。`load_corpus()` 的「同名图片+txt
配对」规则会自动跳过它们。

> 早期归档里封存 test 共 520 帧 / 7 个 unit，其中第 7 个 `imago:t3`（14 帧）
> 来自另一组 imago 测试数据、不属于 gomrade。本仓库只依赖 gomrade，故
> `split.tsv` 中不含它；这里如实记一笔，避免与历史数字对不上。

### 数据为什么不需要"中间产物"

gomrade 的原始文件本身已经包含训练所需的全部信息：

| 训练需要 | 原始文件 | 说明 |
|---|---|---|
| 图片 | `<unit>/*.png` / `*.jpg` | 原样读取，`convert("RGB")` |
| 逐点真值 | 同名 `*.txt` | 19 行 × 19 个 `.`/`B`/`W`，以空格分隔；读入内存去掉空格即可 |
| 四角坐标 | `<unit>/board_extractor_state.yml` 的 `pts_clicks` | 4 个 `[x,y]`，顺序即左上→右上→右下→左下，已是原图像素坐标 |

早期开发时曾把这些信息摊平成 `dataset.json` + `unified/*.txt` + `geometry/*.json`
等中间产物。它们**都是可再生的**，且 `dataset.json` 从来没有生成脚本，开发者无法
照着构造 —— 现已全部移除。`load_corpus()` 直接枚举原始目录，不再依赖任何中间
文件。

**唯一无法从原始文件推出的是 train/val/test 划分**（一次性随机分层的结果），
因此随仓库提供 `training/cnn/split.tsv`：61 行 `unit → 角色` 的纯文本表，约
3 KB，不含任何图片、真值或坐标内容。

---

## 2. 准备数据

从 Kaggle 下载 gomrade 并解压到任意目录，保持原样目录结构：

```
<gomrade 解压目录>/
  dataset/
    1/                       11.png、11.txt、board_extractor_state.yml、...
    11/                      ...
    Hueber-Fanti/            ...
  dataset2/
    13/                      ...
    Grazzini-Bevegni/        ...
```

- `dataset/` 和 `dataset2/` 两个子目录必须都在；不要重命名子目录（unit 名是
  划分表里的键）。
- 不要修改图片、真值 txt 或 yml —— 这会改变 patch 内容，导致回放出的模型与
  签入版本不一致。
- `--data` 直接指向这个含 `dataset/` `dataset2/` 的目录。

---

## 3. 依赖（训练时）

推理侧 `inference/` 是零第三方依赖的原生 ES modules；**只有训练需要装 Python 包**。
以下为签入模型的实际训练环境，版本已固定在 `requirements.txt`：

| 项 | 版本 |
|---|---|
| Python | 3.12.3 |
| torch | 2.15.0+cpu |
| onnx | 1.23.2 |
| numpy | 2.5.4 |
| Pillow | 13.0.0 |
| CPU | 2 核，无 GPU |

一条命令装齐：

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

> 字节级回放依赖上述版本与 CPU 线程数。换 torch/numpy 版本、换 GPU、或改
> `--threads`，都可能让浮点运算顺序变化，从而产出不同的权重字节。

---

## 4. 训练命令

在**仓库根目录**执行（`-m training.cnn.train_cnn` 依赖包相对导入）：

```bash
python3 -m training.cnn.train_cnn \
  --data /path/to/gomrade 解压目录 \
  --out training/out-cnn \
  --epochs 5 --batch 512 --lr 1e-3 --cell 40 --seed 20261105 --threads 2
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--data` | **必填** | §2 的 gomrade 解压目录 |
| `--out` | `training/out-cnn` | 产出目录 |
| `--epochs` | 5 | 训练轮数 |
| `--batch` | 512 | batch size |
| `--lr` | 1e-3 | Adam 学习率 |
| `--cell` | 40 | warp 后每格像素数 |
| `--seed` | 20261105 | 训练种子，改动会改变模型字节 |
| `--threads` | 2 | torch 线程数，为确定性固定 |

**`--cell` 必须与推理侧一致**：`inference/src/cnn_inference.mjs` 的
`CELL_DEFAULT` / `PATCH_SIZE` 和 `train_cnn.py` 的 `PATCH` 都是 40，三处需同步。

流程：`load_corpus` 直接枚举 gomrade → `decode_image` 解码 → `truth_matrix`
解析真值 → 从 yml 读四角 → `warp_perspective` 透视拉正 → 每交叉点切 40×40
patch → LeNet 风格 CNN（3 类 `.`/`B`/`W`）→ 导出 ONNX。

网络与超参固定为
`Conv(1,16,3)-ReLU-MaxPool-Conv(16,32,3)-ReLU-MaxPool-FC(800,64)-FC(64,3)`，
Adam lr=1e-3，5 epoch，batch 512，无 dropout、无数据增强。

### 确定性是怎么保证的

`train_cnn.py` 在任何张量分配之前固定：

- `torch.manual_seed(SEED)` / `np.random.seed(SEED)` / `PYTHONHASHSEED`
- `torch.set_num_threads(2)` / `set_num_interop_threads(1)`
- `torch.use_deterministic_algorithms(True)`
- 文件枚举按 unit 名与文件名排序，patch 顺序固定
- ONNX 导出用常量 dummy（`torch.zeros`）而非 `torch.randn`

日志会打印 `sample order sha256`（样本顺序指纹）和 `cnn.onnx sha256`，
便于核对是不是同一份输入、同一个产出。

---

## 5. 产出与回放校验

全部写到 `--out` 目录。其中 **`cnn.onnx` 已随仓库签入**（demo 直接加载），
其余为中间产物，被 `.gitignore` 屏蔽：

| 文件 | 说明 |
|---|---|
| `cnn.onnx` | **已签入**的推理模型；demo 直接加载 |
| `cnn.pt` | PyTorch `state_dict` 存档，不签入 |
| `cnn_meta.json` | 架构 / 超参 / seed / 样本顺序指纹 / 模型 sha256，不签入 |
| `cnn_eval.json` | 验证集混淆矩阵与 per-class 指标，不签入 |

**校验字节级回放**（先把签入模型备份一下，因为重跑会覆盖它）：

```bash
cp training/out-cnn/cnn.onnx /tmp/cnn.onnx.before
python3 -m training.cnn.train_cnn --data /path/to/gomrade 解压目录 \
  --out training/out-cnn
sha256sum /tmp/cnn.onnx.before training/out-cnn/cnn.onnx
cmp /tmp/cnn.onnx.before training/out-cnn/cnn.onnx && echo "字节一致"
# 或者：git diff --exit-code -- training/out-cnn/cnn.onnx
```

本仓库签入的版本已用**完全相同的命令、在两个独立输出目录里连续重跑两遍**，
两次 `cnn.onnx` 与 `cnn.pt` 均逐字节一致：

```
51b38d3570da058edc7873b855ee34e9a458cc0324b944589f088b2388745326  cnn.onnx  (545915 B)
```

两遍的 `sample order sha256` 同为
`6f2a403c7bfa9e8b2a57cc1278be0aa53ea171d8c6c9420ca35a6f44c67f0f61`，
`val_acc` 同为 `0.9835`。签入文件即第二遍产物。若你的环境版本与上表不同，
`cmp` 可能失败，但训练流程与模型结构仍然一致。

---

## 6. 与 demo 联调

`inference/tools/serve.mjs` 的静态根是**仓库根目录**，默认把
`/training/out-cnn/cnn.onnx` 作为模型 URL。模型已随仓库签入，无需先训练：

```bash
cd inference && node tools/serve.mjs
# 浏览器打开 http://127.0.0.1:8899/inference/app/
# 导入照片 → 点 4 角 → 点「识别 (CNN)」
```

ONNX Runtime Web 运行时从 CDN（jsDelivr，onnxruntime-web@1.19.2）加载，
首次需联网；模型文件本身随仓库分发。
