# 数学题 OCR 批量识别项目

本项目用于完成 `0423OCR识别操作文档.pdf` 中平时作业的 OCR 识别任务。程序会读取数学题图片，批量识别中文题干、选项和数学公式，并生成课程要求的 JSON 结果文件。

## 最终结果

当前最终结果文件：

```text
output/2023215822_jiangshuoyang.json
```

该文件以图片文件名为键，以 OCR 文本为值，格式符合课程文档中的提交要求。

实验报告：

```text
实验报告.md
```

## 技术方案

主模型使用开源 `PaddleOCR-VL 1.6` 完整文档解析 pipeline。该模型适合处理中文、数学公式、表格、选项和版面顺序混合的题目截图。

兜底模型只在主结果出现明显风险时启用：

- `RapidOCR`：用于文本和选项局部修补。
- `Pix2Text`：用于高风险公式样本的局部修补。

项目不会无条件混合多个模型结果，避免为了补字引入新的严格错误。

## 运行环境

推荐使用当前环境：

```text
C:\Users\Administrator\.conda\envs\my_llm-project\python.exe
```

当前项目默认使用 GPU：

```python
DEVICE = "gpu:0"
PIPELINE_VERSION = "v1.6"
```

依赖安装：

```powershell
& 'C:\Users\Administrator\.conda\envs\my_llm-project\python.exe' -m pip install -r requirements.txt
```

PaddlePaddle GPU 版本需要按本机 CUDA 环境单独安装。

## 运行方式

在 PyCharm 中可以直接运行 `main.py`。常用参数位于文件顶部：

```python
LIMIT_IMAGES = 0
SKIP_FIRST_IMAGES = 0
SKIP_EXISTING = True
RESET_OUTPUT_BEFORE_RUN = False
RESET_RAW_OUTPUT_BEFORE_RUN = False
SAVE_EVERY = 20
```

参数含义：

- `LIMIT_IMAGES = 0`：处理全量数据。
- `SKIP_EXISTING = True`：断点续跑，跳过 JSON 中已有结果。
- `RESET_OUTPUT_BEFORE_RUN = False`：不删除已有结果。
- `SAVE_EVERY = 20`：每处理 20 张新图片保存一次。

命令行运行：

```powershell
& 'C:\Users\Administrator\.conda\envs\my_llm-project\python.exe' main.py --limit 0 --skip-existing --save-every 20 --device gpu:0 --pipeline-version v1.6 --vl-mode python
```

重新从头跑时才使用：

```powershell
--reset-output
```

正常续跑不要使用 `--reset-output`。

## 项目结构

```text
main.py                  主入口
src/engines.py           OCR 模型封装
src/runner.py            批量运行、断点续跑、保存结果
src/image_utils.py       图片枚举和保守预处理
src/text_tools.py        文本清洗、公式修复、风险检测
src/json_io.py           JSON 读写
output/                  最终结果文件
实验报告.md              C 项实验报告
```

临时图片、raw 中间输出、缓存和旧评估样例已经清理，不影响 `main.py` 复现当前流程。
