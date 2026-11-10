# weiqi-photo

浏览器端「拍照还原围棋棋局」：导入棋盘照片 → 点选四个角 → CNN 在浏览器本地
把每个交叉点识别为「空 / 黑 / 白」，还原棋子及其坐标矩阵。

实现：在同一份公开标注数据集上，用 `training/cnn/train_cnn.py` 可**字节级回放**
地训练出 CNN 模型；该模型随仓库签入 `training/out-cnn/cnn.onnx`。推理侧用 ONNX
Runtime Web 在浏览器本地加载模型，把四角内的棋盘透视校正后，对每个交叉点切
40×40 灰度 patch 做三分类，得到棋子与坐标。全程不上传服务器。

本仓库包含推理代码、训练代码、已签入的模型和完整的训练说明；**不含训练数据**
（数据从 Kaggle 自行获取，见下）。

## 目录

```
inference/                  浏览器推理（零运行时依赖，原生 ES modules）
  app/index.html            demo 页面
  app/main.js               图片载入、四点选角、调用识别
  src/host.mjs              图片解码（createImageBitmap + canvas）
  src/homography.mjs        透视几何（四角 → 交叉点坐标）
  src/warp.mjs              透视校正
  src/cnn_inference.mjs     ONNX 推理（patch 提取 + argmax）
  src/index.mjs             公共库入口 + transcribe()
  tools/serve.mjs           本地静态服务器（仅 Node 内置模块）
  package.json              npm 脚本
training/                   离线训练（字节级可回放）
  cnn/train_cnn.py          训练入口（固定 seed / 线程数 / 样本顺序）
  cnn/common.py             直读 gomrade 目录：图片 / 真值 / 四角
  cnn/warp.py               训练侧透视校正
  cnn/split.tsv             unit → train/val/test 划分（约 3 KB）
  out-cnn/cnn.onnx          已签入的推理模型（demo 直接加载）
requirements.txt            固定版本的训练依赖
TRAINING.md                 数据来源、致谢、许可、重放步骤
```

## 跑 demo

模型已随仓库签入 `training/out-cnn/cnn.onnx`，无需先训练。ONNX Runtime Web
运行时从 CDN 加载（首次需联网）。

```bash
cd inference && node tools/serve.mjs
# 浏览器打开 http://127.0.0.1:8899/inference/app/
```

导入照片 → 依次点 **左上 → 右上 → 右下 → 左下** 四个可下子区域的角点 →
点「识别」。

## 训练回放

训练数据来自公开的 **gomrade** 数据集（Kaggle: davids1992），本仓库不含数据、
也没有 `dataset.json` 之类的中间产物：`load_corpus()` 直接读取 gomrade 解压目录
里的图片、同名真值 txt 和 `board_extractor_state.yml` 四角信息。完整说明见
`TRAINING.md`。概要：

```bash
pip install -r requirements.txt
python3 -m training.cnn.train_cnn --data <gomrade 解压目录> --out training/out-cnn
```

在 `TRAINING.md` §3 列出的固定版本依赖下，重跑产出的 `cnn.onnx` 与签入版本
**逐字节一致**（脚本固定了 seed、线程数、样本枚举顺序和 ONNX 导出 dummy）。
重跑会覆盖该文件，`git diff -- training/out-cnn/cnn.onnx` 应为空。其余产出
（`cnn.pt`、`cnn_meta.json`、`cnn_eval.json`）为中间产物，被 `.gitignore` 屏蔽。

## 数据来源与致谢

🙏 **特别感谢 [gomrade 数据集](https://www.kaggle.com/datasets/davids1992/gomrade-dataset-go-baduk-images-with-labels)
（作者 davids1992）及所有贡献者。**

本项目签入的 CNN 模型完全由该数据集训练得到：1620 帧（1325 train / 295 val）、
39 个 unit、584820 个 40×40 patch。没有这份公开标注数据，就没有这个模型。
数据来源、划分方式、过滤规则与完整许可说明见 `TRAINING.md` §1。

gomrade 采用 **[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)**：

- 本仓库**不分发**任何 gomrade 文件；训练时只做内存态解析，不生成衍生副本；
- 该数据集及由其训练得到的模型权重**仅限非商业使用**，本仓库的 MIT 许可
  **不适用于**数据部分；
- ND（禁止演绎）条款请以原作者与 Kaggle 页面为准，请勿分发数据的改编版本。

## 许可

- **代码**：MIT（见 `LICENSE`）。运行时依赖 ONNX Runtime Web（微软，MIT）从 CDN
  获取，不在本仓库分发。
- **训练数据与训练产物**：受 gomrade 的 CC BY-NC-ND 4.0 约束，见上节与
  `TRAINING.md` §1。两者许可范围不同，请勿混淆。
