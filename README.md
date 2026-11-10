# weiqi-photo

浏览器端「拍照还原围棋棋局」：导入棋盘照片 → 点选四个角 → CNN 在浏览器本地
把每个交叉点识别为「空 / 黑 / 白」，还原棋子及其坐标矩阵。

实现：在同一份标注数据集上，用 `training/cnn/train_cnn.py` 可回放地训练出 CNN
模型；该模型随仓库签入 `training/out-cnn/cnn.onnx`。推理侧用 ONNX Runtime Web
在浏览器本地加载模型，把四角内的棋盘透视校正后，对每个交叉点切 40×40 灰度
patch 做三分类，得到棋子与坐标。全程不上传服务器。

本仓库包含推理代码、训练代码、已签入的模型和训练说明；不含训练数据，不含
第三方运行时二进制（ONNX Runtime Web 运行时从 CDN 获取）。

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
training/                   离线训练（可回放，产出 CNN 模型）
  cnn/train_cnn.py          训练入口
  cnn/common.py             数据装载 / 几何 / 真值
  cnn/warp.py               训练侧透视校正
  out-cnn/cnn.onnx          已签入的推理模型（demo 直接加载）
requirements.txt            训练依赖
TRAINING.md                 数据格式、依赖、训练重放步骤
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

## 训练重放

数据自备，步骤见 `TRAINING.md`。概要：

```bash
pip install -r requirements.txt
python3 -m training.cnn.train_cnn --data <数据目录> --out training/out-cnn
```

用相同数据集与超参可复现出与签入版本一致的 `cnn.onnx`。重跑会覆盖该文件，
`git diff` 可直接核对产出是否与已签入模型一致。其余产出（`cnn.pt`、
`cnn_meta.json`、`cnn_eval.json`）为中间产物，被 `.gitignore` 屏蔽，不签入。

## 许可

MIT（见 `LICENSE`）。ONNX Runtime Web 为微软 MIT 许可，运行时从 CDN 获取，
不在本仓库分发。
