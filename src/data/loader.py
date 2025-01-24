import pandas as pd
from pathlib import Path
from src.config import RAW_DATA_DIR, DATA_FILES
from src.utils.logger import log_info, log_error, log_success

class OlympicsDataLoader:
    def __init__(self):
        self.raw_data_dir = RAW_DATA_DIR
        
    def check_file_exists(self, filename):
        file_path = self.raw_data_dir / filename
        if not file_path.exists():
            log_error(f"文件不存在: {file_path}")
            raise FileNotFoundError(f"找不到文件: {file_path}")
        return file_path

    def load_medal_counts(self):
        """加载奖牌数据"""
        file_path = self.check_file_exists(DATA_FILES['medal_counts'])
        log_info(f"正在加载奖牌数据: {file_path}")
        try:
            df = pd.read_csv(file_path)
            log_success(f"成功加载奖牌数据: {len(df)} 行")
            return df
        except Exception as e:
            log_error(f"加载奖牌数据失败: {str(e)}")
            raise

    def load_all_data(self):
        """加载所有数据，使用合适的编码"""
        data = {}
        encoding_map = {
            'athletes': 'utf-8',
            'medal_counts': 'utf-8',
            'hosts': 'utf-8',
            'programs': 'cp1252',  # 修改programs文件的编码
            'dictionary': 'cp1252'  # 修改dictionary文件的编码
        }

        for key, filename in DATA_FILES.items():
            try:
                file_path = self.check_file_exists(filename)
                encoding = encoding_map.get(key, 'utf-8')
                log_info(f"正在加载 {key} 数据 (编码: {encoding})")
                
                data[key] = pd.read_csv(
                    file_path,
                    encoding=encoding,
                    on_bad_lines='skip'  # 添加错误行处理
                )
                log_success(f"成功加载 {key} 数据: {len(data[key])} 行")
            except Exception as e:
                log_error(f"加载 {key} 数据失败: {str(e)}")
                raise

        return data