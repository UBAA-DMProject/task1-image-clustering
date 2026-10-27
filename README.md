# 基于颜色统计与方向梯度特征的图像无监督聚类比较研究

**数据挖掘导论 2026 大作业 · 任务1**

本研究以课程提供的600张图像为对象，比较两种特征表示对图像聚类的影响。方法A采用三通道平均颜色，方法B采用方向梯度直方图（HOG）；两种方法均使用相同配置的K-Means。实验包括分方法特征提取与模型拟合、统一评价、类别混淆分析和随机初始化稳定性检验。

完整实现与学术论述位于 [image_clustering.ipynb](image_clustering.ipynb)。Notebook已保存实际运行输出，可查看图像、指标表、结果分析和参考文献。

## 1. 研究设计

设图像特征为 `X`，K-Means在预设簇数 `K=6` 下最小化组内平方距离和。类别数量来自题目先验，样本的真实标签只在两种方法的全部模型训练完成后用于评价。

两种方案的**聚类算法相同，差异在于特征表示及其预处理**。本实验未训练有监督分类器，也未使用预训练神经网络。

| 项目 | 方法A：平均颜色＋K-Means | 方法B：HOG＋K-Means |
|---|---|---|
| 输入处理 | 转换为RGB，保留原始分辨率 | 转为灰度，Lanczos缩放为128×128 |
| 特征定义 | 三个颜色通道的全局均值 | 局部梯度方向直方图 |
| 归一化 | 无额外缩放；三个通道均为0～255 | 像素缩放至0～1；L2-Hys块归一化 |
| 特征维度 | 3 | 1764 |
| 特征矩阵变量 | `X_rgb` | `X_hog` |
| 重复训练结果 | `rgb_runs` | `hog_runs` |
| 主实验模型 | `rgb_primary` | `hog_primary` |
| Notebook章节 | 第3节 | 第4节 |
| 独立结果目录 | `results/method_a_mean_rgb/` | `results/method_b_hog/` |

### 1.1 方法A：平均颜色统计

对原始图像的红、绿、蓝通道分别计算像素均值，构成三维向量。该基线用于分析全局颜色统计的区分能力。其表达不保留像素位置，可能合并整体颜色相似的不同物体，或将颜色变化较大的同类图像拆分。

### 1.2 方法B：HOG表示

图像统一为128×128灰度图，每个16×16像素单元统计9个方向区间，相邻2×2单元组成归一化块，使用L2-Hys归一化，不启用平方根变换，也不额外进行特征列标准化。重叠块描述符拼接后得到1764维向量，表征局部轮廓与纹理分布。

### 1.3 公共聚类协议

| 参数 | 设置 |
|---|---|
| 算法与距离 | K-Means；欧氏距离；K=6 |
| 初始化 | k-means++；n_init=10 |
| 迭代与停止 | Lloyd更新；max_iter=300；tol=1e-4 |
| 主要实验 | 固定random_state=42 |
| 稳定性实验 | 固定种子0、1、7、21、42，报告全部结果 |
| 外部评价 | 最佳一对一匹配后的ACC、ARI、NMI |
| 内部评价 | 原始特征空间中的轮廓系数 |

`n_init=10` 表示每个随机种子尝试十次初始化，按K-Means目标函数选择结果，不依据真实标签选择。PCA仅用于二维展示，不参与训练。不同特征空间的K-Means目标函数值不直接用于比较。

## 2. Notebook组织

| 章节 | 内容 |
|---|---|
| 摘要、研究问题 | 研究目标、符号定义与K-Means目标函数 |
| 第2节：公共实验协议 | 环境、数据索引、公共参数、训练与展示函数 |
| 第3节：方法A | 原理与预处理 → 颜色特征提取 → 独立训练 → 簇规模及样本 |
| 第4节：方法B | 原理与预处理 → HOG特征提取 → 独立训练 → 簇规模及样本 |
| 第5节：统一评价 | 标签对齐、评价定义、方法A结果、方法B结果 |
| 第6节：对照实验 | 主要指标、类别—簇交叉表、随机种子实验与PCA |
| 第7～8节：讨论与结论 | 混淆模式、内部指标解释、适用范围与复现记录 |
| 第9节：参考文献 | 方法原始文献与软件实现文档 |

## 3. 环境配置

需要Python 3.12或更高版本；本次实际验证环境为 **Python 3.14.7、Windows**。直接依赖版本固定于 `requirements.txt`。

在本仓库目录打开终端，创建独立环境并安装依赖。Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

macOS / Linux：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## 4. 数据准备与运行

将课程提供的 `Cluster.zip` 放入 `data/Cluster.zip`，无需解压。压缩包内部结构应为：

```text
Cluster/
├── dataset/
│   ├── 001.png
│   └── ...
└── cluster_labels.json
```

也可在启动内核前设置 `CLUSTER_DATA_PATH` 指向其他位置的压缩包；相对路径以仓库根目录为起点。原始数据不包含在上传包中，并通过 `.gitignore` 排除。

运行步骤：

1. 用VS Code打开仓库文件夹，安装Microsoft的Python、Jupyter扩展。
2. 打开 `image_clustering.ipynb`，选择已安装依赖的 `.venv` 环境作为内核。
3. 执行“重启内核”，然后“全部运行”，或按Shift+Enter从上到下逐格执行。
4. 在Notebook中查看结果；各方法及汇总输出分别保存到下述目录。

若使用浏览器界面，可在同一环境额外安装JupyterLab后启动。代码使用相对路径，不依赖作者机器的绝对目录；执行耗时随运行环境变化。

## 5. 实验结果

主要结果固定使用随机种子42：

| 方法 | ACC | ARI | NMI | 轮廓系数 | 匹配后错误数 |
|---|---:|---:|---:|---:|---:|
| A：平均颜色＋K-Means | 77.83% | 0.7728 | 0.8941 | 0.7468 | 133 |
| B：HOG＋K-Means | 100.00% | 1.0000 | 1.0000 | 0.4926 | 0 |

在五个固定种子下，方法A的ACC均为77.83%，方法B均为100.00%；完整逐次结果见 `results/comparison/seed_stability.csv`。

类别—簇交叉表显示，颜色方案合并了皮革与晶体管，同时将瓷砖拆分至多个簇；HOG方案在当前数据中恢复了给定类别划分。颜色方案的轮廓系数反而较高，表明内部几何分离与语义类别一致性并不等价，且两种特征空间的距离结构不可混为一谈。

两方案还同时存在灰度化、分辨率、维度及归一化差异，因此实验支持的是**整体表示方案的比较**，不能将提升单独归因于HOG的某一个环节。当前评价针对全部600张给定图像，未设置独立测试集；这些指标不能解释为新图像上的分类准确率。背景、旋转和光照变化的影响尚未单独验证。

## 6. 输出文件与目录

```text
results/
├── dataset_samples.jpg
├── experiment.json              # 数据摘要、环境、配置、指标与匹配关系
├── analysis.md                  # 由实际结果生成的讨论
├── method_a_mean_rgb/
│   ├── cluster_assignments.csv   # 方法A原始簇指派
│   ├── cluster_sizes.csv
│   ├── cluster_samples.jpg
│   ├── metrics.json
│   ├── contingency.csv
│   └── seed_stability.csv
├── method_b_hog/                # 与方法A相同的文件结构
└── comparison/
    ├── metrics.csv
    ├── cluster_assignments.csv
    ├── sample_evaluation.csv    # 按文件名逐图核对
    ├── seed_stability.csv
    ├── stability_summary.csv
    ├── contingency_matrices.png
    └── pca_projection.png
```

簇编号仅为组名，ACC计算前先求解簇与类别的最佳一对一匹配；该过程只解释分组，不移动样本。交叉统计图保留原始簇编号，非对角计数不直接等于错误。

## 7. 参考与提交范围

方法原理、评价定义和实现接口的编号引用见Notebook第9节，涵盖Lloyd、Dalal与Triggs、Hubert与Arabie、Rousseeuw的原始文献，以及scikit-learn、scikit-image和SciPy官方文档。
