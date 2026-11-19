# CrownRoute — 框监督荔枝树冠实例分割

[English](README.md)

本仓库提供荔枝单株树冠实例分割论文的配套代码。CrownRoute 利用无人机拍摄的果园 RGB 影像和树冠标注框构建训练掩码，对可靠的候选掩码进行细化，必要时根据框内像素重建树冠掩码。论文还比较了使用不同掩码组合训练实例分割模型时，分割性能在训练过程中如何变化。

## 在 CPU 上运行监督构建

此演示已在 Python 3.12 环境下验证。请在仓库根目录运行以下命令，使用两个新目录分别存放合成输入和构建结果。五个弱框分别展示 Final-A、Final-B 及三种进入 Final-C 的情形。

```sh
python -m pip install -r configs/requirements-c1.txt
python -m scripts.make_c1_demo demo-input
python -m scripts.build_supervision --manifest demo-input/patches.json --output demo-output
python -m unittest discover -s tests -p test_c1.py
```

`demo-output/meta/route_outcomes.json` 记录五个输入实例的去向。`demo-output/split/train.txt` 列出产生标准监督的两个实例。使用自备图块和本地 FastSAM-s 权重时，输入格式、命令及掩码坐标约定见 [C1 说明](docs/c1-sources.md)。

## 学生训练、评价与 TSU

| 任务 | 从仓库根目录调用的入口 | 输入 |
| --- | --- | --- |
| C1 固定成员学生比较 | `python student/train_c1.py --help` | 实例记录、成员清单、验证参考与模型初始化 |
| C2 标准配置及边界放宽 | `python student/train_c2.py --help` | 对应冻结成员与统一的验证坐标 |
| C2 固定规模替换 | `python student/train_fixed_n.py --help` | 固定规模成员与同一训练预算 |
| 表 4 选定模型的统一回放 | `python student/evaluate_table4.py --help` | 修正后的 COCO 验证参考与已保存的分割预测，使用官方 COCOeval |
| 表 6 果园清查指标 | `python student/agricultural_metrics.py --help` 导出逐图、逐种子和三种子汇总，单次运行也可在表 4 入口追加 `--inventory` | 按置信度筛选后最大化总掩码 IoU，再保留 IoU ≥ 0.50 的匹配对 |
| 训练中的逐点评价 | `python student/evaluate.py --help` | 单次验证的预测，使用训练期 final5b 评价程序 |
| 四项 TSU 读数 | `python tsu/summarize.py --help` | 三种配置、三个种子的逐坐标验证轨迹 |

C1 固定成员比较按八轮训练。C2 比较按 27,422 次成功更新训练，各次运行共享 47 个实际观察点。精确复现 C2 需获得授权的原始图像、冻结成员清单、修正后的人工验证参考及相同的模型初始化。三种配置的组装方式见 [C2 说明](docs/c2-configurations.md)。

学生训练及验证所需的环境、参数与文件布局见[学生与 TSU 说明](docs/student-and-tsu.md)。配置好相应环境后，执行 `python student/smoke_cpu.py` 可用合成输入检查一次网络前向及完美预测、空预测的掩码 AP。执行 `python student/smoke_matching.py` 可核对重复预测和存在多个可匹配树冠时的不同评价规则。CPU 合成检查仅验证这些组件；论文数值的来源是原正式运行和保存的预测。

补充材料 S11 节的不确定性估计见 [Bootstrap 分析说明](docs/s11-bootstrap.md)。

拿到论文的两份补充 CSV 后，将它们放到当前目录，用 Python 标准库即可复算全部端点。脚本先在单次训练内计算四项读数，再汇总三个种子的均值与样本标准差。

```sh
python tsu/summarize.py Supplementary_Data_S1_Trajectories.csv --verify Supplementary_Data_S2_Endpoints.csv
```

## 数据和版本

本仓库提供论文配套源码和合成示例。论文全文、原始航片、人工标注、完整训练监督集及模型权重由研究团队另行保存。第三方组件及其使用条款见[发布范围说明](docs/distribution.md)。
