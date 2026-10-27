# 数据放置说明

将课程提供的 `Cluster.zip` 放在本目录：

```text
data/
├── README.md
└── Cluster.zip
```

无需解压。Notebook 直接读取压缩包内的 `Cluster/dataset/*.png`，只在评估阶段读取 `Cluster/cluster_labels.json`。

数据文件由课程提供，不包含在代码提交包中，也不应随仓库上传。`.gitignore` 已排除原始数据。

也可以在运行内核之前设置 `CLUSTER_DATA_PATH` 环境变量，指向另一个位置的 `Cluster.zip`；相对路径以仓库根目录为起点。
