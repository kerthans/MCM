# MCM

架构设计

```
olympic_analysis/
│
├── data/                      # 数据文件夹
│   ├── raw/                  # 原始CSV文件
│   │   ├── data_dictionary.csv   
│   │   ├── summerOly_athletes.csv   
│   │   ├── summerOly_hosts.csv   
│   │   ├── summerOly_medal_counts.csv  
│   │   └── summerOly_programs.csv  
│   └── processed/            # 处理后的数据
│
├── src/                      # 源代码
│   ├── __init__.py
│   ├── config.py            # 配置文件
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py       # 数据加载
│   │   └── preprocessor.py  # 数据预处理
│   │
│   ├── analysis/
│   │   └── explorer.py  # 数据分析
│   ├── features/
│   │   ├── __init__.py
│   │   ├── builders.py     # 特征构建
│   │   └── selector.py     # 特征选择
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py        # 基础模型类
│   │   ├── time_series.py # 时间序列模型
│   │   ├── ml_models.py   # 机器学习模型
│   │   ├── predictor.py   
│   │   └── ensemble.py    # 模型集成
│   │
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py     # 评估指标
│       ├── logger.py   
│       └── visualization.py# 可视化工具
│
├── notebooks/               # Jupyter notebooks
│   ├── 1.0-data-exploration.ipynb
│   ├── 2.0-feature-engineering.ipynb
│   └── 3.0-modeling.ipynb
│
├── tests/                  # 单元测试
├── requirements.txt        # 依赖包
└── main.py                # 主运行文件
```
