# CrownRoute

CrownRoute 从 RGB 图块和树冠弱框构建用于实例分割训练的掩码。一张图块上的多个框共用 FastSAM-s 候选池。可靠候选经细化后进入 Final-A；具备重建资格的记录从框内 RGB 重建，通过固定接纳条件后进入 Final-B。C2 使用独立训练的验证轨迹比较监督配置的效用。

## 在 CPU 上运行监督构建

从仓库根目录执行以下命令。用一个新目录生成合成图块、弱框与候选。演示只依赖 Python 3.10+、NumPy 和 OpenCV，能够展示 A、B 以及三种进入 C 的情形。

```sh
python -m pip install -r configs/requirements-c1.txt
python -m scripts.make_c1_demo demo-input
python -m scripts.build_supervision --manifest demo-input/patches.json --output demo-output
python -m unittest discover -s tests -p test_c1.py
```

`demo-output/meta/route_outcomes.json` 记录五个输入实例的去向。`demo-output/split/train.txt` 列出产生标准监督的两个实例。上述依赖清单用于 Python 3.12 环境。使用自备图块和本地 FastSAM-s 权重时，输入格式、命令及掩码坐标约定见 [C1 说明](docs/C1来源.md)。

## 学生训练、评价与 TSU

| 任务 | 从仓库根目录调用的入口 | 输入 |
| --- | --- | --- |
| C1 固定成员学生比较 | `python student/train_c1.py --help` | 实例记录、成员清单、验证参考与模型初始化 |
| C2 标准配置及边界放宽 | `python student/train_c2.py --help` | 对应冻结成员与统一的验证坐标 |
| C2 固定规模替换 | `python student/train_fixed_n.py --help` | 固定规模成员与同一训练预算 |
| 验证预测评价 | `python student/evaluate.py --help` | COCO 验证参考与分割预测 |
| 四项 TSU 读数 | `python tsu/summarize.py --help` | 三种配置、三个种子的逐坐标验证轨迹 |

学生训练及验证所需的环境、参数与文件布局见[学生与 TSU 说明](docs/学生与TSU来源.md)。在该环境内执行 `python student/smoke_cpu.py`，可用合成输入检查一次网络前向及完美预测、空预测的掩码 AP。C1 固定成员比较按八轮训练。C2 比较按 27,422 次成功更新训练，各次运行共享 47 个实际观察点。

拿到论文的两份补充 CSV 后，将它们放到当前目录，用 Python 标准库即可复算全部端点。脚本先在单次训练内计算四项读数，再汇总三个种子的均值与样本标准差。

```sh
python tsu/summarize.py Supplementary_Data_S1_Trajectories.csv --verify Supplementary_Data_S2_Endpoints.csv
```

## 数据和版本

本版本供论文审阅。仓库包含作者项目的计算代码、固定参数和合成示例。论文的完整数值依赖受管理的原始影像、标注和成员清单。各组件的使用条件及公开前需确认的事项见[发布范围](docs/发布范围.md)。
