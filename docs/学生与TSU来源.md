# 学生训练、验证与 TSU

## 入口与论文对应关系

| 论文环节 | 入口 | 运行依据与产物 |
| --- | --- | --- |
| 固定成员路径比较的学生训练 | `student/train_c1.py` | R2_IDFIX 发布包的 `07_LAUNCH_SCRIPTS/run_one.py` 以 B0 普通损失调用 `02_CODE/p06_runner.py`，每次 8 轮，batch 为 2，种子为 42、43、44。原始 `checkpoint_metrics.csv` 对应每次观测，严格高于先前最佳验证 Mask AP 时更新检查点。 |
| 标准及边界放宽配置的时序训练 | `student/train_c2.py` | T28 的 `fresh_panel_runner.py` 经 T31 的六次正式运行验证。标准配置 6,857 条，边界放宽配置 8,009 条。各次运行的 `checkpoint_metrics.csv` 进入 S1。 |
| 固定规模配置的时序训练 | `student/train_fixed_n.py` | E2 最终归档的 `COMMON_PROTOCOL/RUNNERS/e2_fixed_n_runner.py`。保留 4,778 条 A 和 927 条原 B，加入 1,152 条固定恢复记录。每个种子的 `RAW_E2` 记录对应 S1 中 Fixed-N 的 47 个观测点。 |
| 学生模型与人工验证 | `student/model.py`、`student/loss.py`、`student/validation.py`、`student/evaluate.py` | 模型来自 `model_training_snapshot.build_model` 的无额外注意力分支，损失来自 `p06_loss` 的 B0 分支，保留其张量运算。评价采用原项目 final4 / final5 / final5b 与 `P06Evaluator` 的实际验证函数。保存的 COCO 分割预测可直接交给 evaluate。 |
| 四项 TSU | `tsu/summarize.py` | 从正式 E2 归档 `SOURCE_METHODS/analyze_phase_a.py` 的 `analyze` 与 `aggregate_e2.py` 提取，输出格式对应 Supplementary Data S2。S1 的三种配置、每种三个种子、每次 47 个观测点分别计算，再跨种子求均值与样本标准差。 |

C2 使用同一份 `student/eval_steps_47.json`，训练持续到 27,422 次成功优化器更新。每张图像含一个训练记录中的实例目标。学生采用 COCO_V1 初始化的双类 Mask R-CNN ResNet-50-FPN，输入尺度 512，batch 为 2，SGD 学习率 0.005、动量 0.9、权重衰减 0.0005，MultiStepLR 里程碑 20、35，γ=0.1，CUDA AMP。验证使用 235 个图块、888 个人工实例。C1 固定成员对照的预算为八轮。其他历史比较依其各自原有的训练与选模合同解释。

## 输入与运行

以下命令从仓库根目录运行。命令中的 data、evaluation、runs 是示例路径。需自行提供有权使用的影像、掩码、人工验证参考以及 torchvision 原始 COCO_V1 权重。

- `data/standard/meta/dataset_records.json` 是记录对象数组。每条含 `id`、`image_path`、`mask_path`、`patch_image`、`class_type`。图像路径和单实例二值掩码路径可相对于数据集根目录，也可为当前计算机的绝对路径。训练框从正掩码的像素范围导出，右、下边界加 1。
- `data/standard/split/train.txt`、`val.txt` 是有序记录 ID 清单，每行一条。C2 的三套归档数据各用自己的成员清单，两个成功更新入口会检查冻结记录与成员顺序。路径可重定位；其余归档记录字段、经修正的验证参考和观测坐标需要对应正式版本。单独调用 C1 入口可用自备同格式数据，所得结果属于运行者的数据集。
- `evaluation/manual_gt_assets_final5b_repaired/normalized_gt_val_manual_coco.json` 是修正人工参考的 COCO 文件，图像项有 `id`、`patch_stem`、`file_name`、尺寸，实例项的 segmentation 为 RLE。相邻的 `evaluation/manual_gt_assets/gt_val_samples.json` 列出验证样本的 `id`、`patch_stem`、`file_name`；可选的 `patch_image_map.json` 按 patch_stem 映射 RGB 路径。该目录中路径可相对该目录。使用全部验证图块时应保留没有实例的图块。
- 权重缓存结构为 `<TORCH_HOME>/hub/checkpoints/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth`。示例使用 Linux shell，`TORCH_HOME` 指向已有缓存。采用不同种子时为每次训练指定独立输出目录，避免误用上一次运行的续训文件。
- 正式时序运行环境记录为 Python 3.12.13、PyTorch 2.5.1+cu124、torchvision 0.20.1+cu124、pycocotools 2.0.10 和 NVIDIA RTX 4090。学生代码还使用 NumPy 与 Pillow。TSU 汇总仅使用 Python 标准库。

```sh
export TORCH_HOME=/path/to/your/torch-cache
WEIGHT="$TORCH_HOME/hub/checkpoints/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"
MANUAL=evaluation/manual_gt_assets_final5b_repaired
python student/train_c1.py --dataset data/standard --manual-root "$MANUAL" --output runs/c1_standard_42 --seed 42
python student/train_c2.py --arm B0 --dataset data/standard --manual-root "$MANUAL" --init-weight "$WEIGHT" --output runs/c2_standard_42 --seed 42
python student/train_c2.py --arm RECOVERED_C --dataset data/boundary_relaxed --manual-root "$MANUAL" --init-weight "$WEIGHT" --output runs/c2_relaxed_42 --seed 42
python student/train_fixed_n.py --arm FIXED_N --dataset data/fixed_n --manual-root "$MANUAL" --init-weight "$WEIGHT" --output runs/c2_fixed_42 --seed 42
python student/evaluate.py --gt "$MANUAL/normalized_gt_val_manual_coco.json" --predictions saved_segm_predictions.json
python tsu/summarize.py Supplementary_Data_S1_Trajectories.csv --verify Supplementary_Data_S2_Endpoints.csv
```

验证程序用最终保存的预测列表回算 Mask AP，输入与对应验证图像 ID 一一匹配。训练中的预测按原协议保留分数不低于 0.05 的实例，每图最多 100 个。构建二值预测掩码使用 0.5 概率阈值。边框 AP 调用 pycocotools，掩码 AP 调用项目实际采用的 final5b 安全解码及 NumPy 匹配/积分函数。损失诊断使用的伪验证样本与选模使用的人工参考分别读取。

## 计算与验证范围

Peak 取单次运行最高 Mask AP；Top-5 取最高五个观测的均值；nAUC 按实际观测坐标作梯形积分，除以首末观测间的跨度；Late 对成功更新进度达到 0.75 的观测取均值。输入 mask_ap 为 0–1 的比例，S2 同时报告比例与 AP 点。九次运行的 423 个观测在本地复算，S2 的 36 个逐种子值、24 个比例尺度均值和标准差、24 个 AP 点尺度均值和标准差均与原表完全相同。

运行 `python student/smoke_cpu.py` 可在 CPU 上检查合成样例。此次以合成图像实际读取相对路径的记录、重建训练框、载入 final5b 验证输入。完整匹配与空预测的 Mask AP 分别为 1 和 0，无预训练权重的学生网络完成一次前向推理，论文分支的 ROI 掩码损失与原归档实现逐值一致。此次没有运行新的正式训练。实际检查使用 Python 3.12.14、torch 2.12.0、torchvision 0.27.0、pycocotools 2.0.11；历史训练环境见上文。

本仓库保留论文使用的普通损失及实际模型和评价路径，清理了归档脚本中未启用的实验选项、机器路径和历史入口。历史 C1 记录曾把 B 权重字段写为 0.7；该字段在实际 B0 损失中未启用。当前 C1 输出记录有效权重为 1.0。模型初始化、损失、采样、优化与评价方法按原计算保留。C2 输入检查将特定机器的文件路径与科学记录字段分开核对。原始图像、标签、预测、成员清单、模型权重和依赖源码均由其持有方管理。第三方软件与权重按各自许可证和获取条件使用，向外发布时还须由权利人确认项目源码的授权。