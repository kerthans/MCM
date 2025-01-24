from pathlib import Path

# 路径配置
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# 创建必要的目录
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# 数据文件配置
DATA_FILES = {
    'athletes': 'summerOly_athletes.csv',
    'medal_counts': 'summerOly_medal_counts.csv',
    'hosts': 'summerOly_hosts.csv',
    'programs': 'summerOly_programs.csv',
    'dictionary': 'data_dictionary.csv'
}


# # 数据类型定义
# MEDAL_COUNTS_DTYPES = {
#     'Rank': 'Int64',
#     'NOC': 'str',
#     'Gold': 'Int64',
#     'Silver': 'Int64',
#     'Bronze': 'Int64',
#     'Total': 'Int64',
#     'Year': 'Int64'
# }

# ATHLETES_DTYPES = {
#     'Name': 'str',
#     'Sex': 'category',
#     'Team': 'str',
#     'NOC': 'str',
#     'Year': 'Int64',
#     'City': 'str',
#     'Sport': 'str',
#     'Event': 'str',
#     'Medal': 'category'
# }

# HOSTS_DTYPES = {
#     'Year': 'Int64',
#     'Host': 'str'
# }

# # 配置参数
# RANDOM_STATE = 42