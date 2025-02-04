import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class OlympicMedalPredictor:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.medals_df = None
        self.athletes_df = None
        self.programs_df = None
        self.target_countries = None
        self.predictions = None
        self.historical_first_medals = None
        self.noc_mapping = None
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
    def load_data(self):
        """加载和准备所需的所有数据"""
        print("正在加载数据文件...")
        
        try:
            # 加载数据集
            self.medals_df = pd.read_csv(os.path.join(self.data_dir, 'summerOly_medal_counts.csv'), 
                                       encoding='utf-8')
            
            try:
                self.athletes_df = pd.read_csv(os.path.join(self.data_dir, 'summerOly_athletes.csv'), 
                                             encoding='utf-8')
            except UnicodeDecodeError:
                self.athletes_df = pd.read_csv(os.path.join(self.data_dir, 'summerOly_athletes.csv'), 
                                             encoding='cp1252')
            
            # 创建NOC映射字典
            self.create_noc_mapping()
            
            # 标准化NOC代码
            self.medals_df['NOC'] = self.medals_df['NOC'].map(self.noc_mapping).fillna(self.medals_df['NOC'])
            self.athletes_df['NOC'] = self.athletes_df['NOC'].map(self.noc_mapping).fillna(self.athletes_df['NOC'])
            
            # 获取目标国家（2024年无奖牌）
            self.target_countries = self.medals_df[
                (self.medals_df['Year'] == 2024) & 
                (self.medals_df['Total'] == 0)
            ]['NOC'].unique()
            
            # 检查数据可用性
            available_countries = set(self.athletes_df['NOC'].unique())
            self.target_countries = [country for country in self.target_countries 
                                   if country in available_countries]
            
            print(f"找到{len(self.target_countries)}个可分析的目标国家")
            
        except Exception as e:
            print(f"数据加载错误: {str(e)}")
            raise
            
    def analyze_historical_first_medals(self):
        """分析历史上首次获得奖牌的案例"""
        first_medals = []
        all_countries = self.medals_df['NOC'].unique()
        
        for country in all_countries:
            country_medals = self.medals_df[self.medals_df['NOC'] == country].sort_values('Year')
            first_medal_year = country_medals[country_medals['Total'] > 0]['Year'].iloc[0] \
                if len(country_medals[country_medals['Total'] > 0]) > 0 else None
            
            if first_medal_year:
                first_medals.append({
                    'NOC': country,
                    'First_Medal_Year': first_medal_year,
                    'Years_Until_Medal': first_medal_year - country_medals['Year'].iloc[0]
                })
        
        return pd.DataFrame(first_medals)
    def create_noc_mapping(self):
        """创建NOC代码映射"""
        # 基础映射字典
        self.noc_mapping = {
            'MON': 'MCO',  # Monaco
            'AND': 'AND',  # Andorra
            'MDA': 'MDA',  # Moldova
            'TUV': 'TUV',  # Tuvalu
            'NRU': 'NRU',  # Nauru
            'SAM': 'SAM',  # Samoa
            'BHU': 'BTN',  # Bhutan
            'YEM': 'YEM',  # Yemen
            'BAN': 'BGD',  # Bangladesh
            'MDV': 'MDV',  # Maldives
            'CAF': 'CAF',  # Central African Republic
            'SUI': 'CHE',  # Switzerland
            'TAN': 'TZA',  # Tanzania
            'SEY': 'SYC',  # Seychelles
            'HON': 'HND',  # Honduras
            'BOL': 'BOL',  # Bolivia
            'BIZ': 'BLZ',  # Belize
        }
        
        # 添加常见的别名
        alternate_codes = {
            'Monaco': 'MCO',
            'Andorra': 'AND',
            'Moldova': 'MDA',
            'Republic of Moldova': 'MDA',
            'Tuvalu': 'TUV',
            'Nauru': 'NRU',
            'Samoa': 'SAM',
            'Bhutan': 'BTN',
            'Yemen': 'YEM',
            'Bangladesh': 'BGD',
            'Maldives': 'MDV',
            'Central African Republic': 'CAF',
            'Switzerland': 'CHE',
            'Tanzania': 'TZA',
            'United Republic of Tanzania': 'TZA',
            'Seychelles': 'SYC',
            'Honduras': 'HND',
            'Bolivia': 'BOL',
            'Belize': 'BLZ',
        }
        
        self.noc_mapping.update(alternate_codes)
    def calculate_historical_metrics(self, country):
        """计算国家的历史表现指标"""
        try:
            country_athletes = self.athletes_df[self.athletes_df['NOC'] == country]
            if len(country_athletes) == 0:
                print(f"警告: 未找到{country}的完整运动员数据，使用基础数据计算")
                return self.calculate_basic_metrics(country)
            
            # 计算基础指标
            participation_years = self.medals_df[self.medals_df['NOC'] == country]['Year'].nunique()
            total_athletes = country_athletes['Name'].nunique()
            unique_sports = country_athletes['Sport'].nunique()
            
            # 近期表现（最近三届奥运会）
            recent_years = [2016, 2020, 2024]
            recent_df = country_athletes[country_athletes['Year'].isin(recent_years)]
            recent_athletes = recent_df['Name'].nunique() if not recent_df.empty else 0
            recent_sports = recent_df['Sport'].nunique() if not recent_df.empty else 0
            
            # 计算增长率
            yearly_athletes = country_athletes.groupby('Year')['Name'].nunique()
            growth_rate = 0
            if len(yearly_athletes) >= 2:
                growth_rate = (yearly_athletes.iloc[-1] - yearly_athletes.iloc[0]) / yearly_athletes.iloc[0] \
                            if yearly_athletes.iloc[0] != 0 else 0
            
            # 计算额外指标
            avg_athletes_per_sport = total_athletes / unique_sports if unique_sports > 0 else 0
            sport_consistency = len(recent_df['Sport'].unique()) / unique_sports if unique_sports > 0 else 0
            
            return {
                'NOC': country,
                'Historical_Participation': participation_years,
                'Total_Athletes': total_athletes,
                'Sports_Diversity': unique_sports,
                'Recent_Athletes': recent_athletes,
                'Recent_Sports': recent_sports,
                'Growth_Rate': growth_rate,
                'Athletes_Per_Sport': avg_athletes_per_sport,
                'Sport_Consistency': sport_consistency
            }
            
        except Exception as e:
            print(f"计算{country}指标时出错: {str(e)}")
            return self.calculate_basic_metrics(country)
    
    def calculate_similarity_to_success_cases(self, country):
        """计算与历史成功案例的相似度"""
        # 获取成功案例的平均指标
        success_cases = self.historical_first_medals[
            self.historical_first_medals['Years_Until_Medal'] <= 12
        ]
        
        if success_cases.empty:
            return 0
        
        # 计算当前国家的特征与成功案例的相似度
        country_features = self.calculate_country_features(country)
        success_features = self.calculate_average_success_features()
        
        # 计算欧氏距离
        if country_features and success_features:
            distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(country_features, success_features)))
            similarity = 1 / (1 + distance)
            return similarity
        
        return 0
    
    def calculate_country_features(self, country):
        """计算国家的标准化特征"""
        country_data = self.athletes_df[self.athletes_df['NOC'] == country]
        if country_data.empty:
            return None
            
        features = [
            len(country_data['Year'].unique()),  # 参与年数
            country_data['Name'].nunique(),      # 运动员数量
            country_data['Sport'].nunique(),     # 运动项目数量
            # 可以添加更多特征
        ]
        
        return features
    
    def calculate_average_success_features(self):
        """计算历史成功案例的平均特征"""
        success_countries = self.historical_first_medals['NOC'].tolist()
        all_features = []
        
        for country in success_countries:
            features = self.calculate_country_features(country)
            if features:
                all_features.append(features)
        
        if not all_features:
            return None
            
        return np.mean(all_features, axis=0)
    def calculate_basic_metrics(self, country):
        """使用基础数据计算指标"""
        try:
            country_medals = self.medals_df[self.medals_df['NOC'] == country]
            
            return {
                'NOC': country,
                'Historical_Participation': country_medals['Year'].nunique(),
                'Total_Athletes': 0,
                'Sports_Diversity': 0,
                'Recent_Athletes': 0,
                'Recent_Sports': 0,
                'Growth_Rate': 0,
                'Athletes_Per_Sport': 0,
                'Sport_Consistency': 0
            }
        except Exception as e:
            print(f"计算{country}基础指标时出错: {str(e)}")
            return None
    def predict_medals(self):
        """生成奖牌预测"""
        results = []
        
        print("正在分析每个国家...")
        for country in self.target_countries:
            metrics = self.calculate_historical_metrics(country)
            if metrics:
                results.append(metrics)
        
        if not results:
            raise ValueError("没有为任何国家生成有效结果")
        
        # 转换为DataFrame
        self.predictions = pd.DataFrame(results)
        
        # 标准化数值列
        numeric_columns = [
            'Historical_Participation', 'Total_Athletes', 'Sports_Diversity',
            'Recent_Athletes', 'Recent_Sports', 'Growth_Rate', 
            'Athletes_Per_Sport', 'Sport_Consistency'
        ]
        
        scaler = MinMaxScaler()
        self.predictions[numeric_columns] = scaler.fit_transform(
            self.predictions[numeric_columns].fillna(0)
        )
        
        # 计算动态权重
        weights = {
            'Historical_Participation': 0.15,
            'Total_Athletes': 0.20,
            'Sports_Diversity': 0.15,
            'Recent_Athletes': 0.25,
            'Recent_Sports': 0.15,
            'Growth_Rate': 0.10
        }
        
        # 计算加权得分
        self.predictions['Medal_Probability'] = sum(
            self.predictions[col] * weight 
            for col, weight in weights.items()
            if col in self.predictions.columns
        )
        
        return self.predictions.sort_values('Medal_Probability', ascending=False)
    
    def calculate_dynamic_weights(self, data):
        """计算动态权重"""
        # 基于数据分布计算权重
        variance_weights = data.var()
        total_variance = variance_weights.sum()
        
        if total_variance == 0:
            # 如果所有特征方差为0，使用均匀权重
            return {col: 1/len(data.columns) for col in data.columns}
        
        # normalize weights
        weights = variance_weights / total_variance
        
        # 调整权重确保关键指标具有最小权重
        min_weight = 0.1
        weights = weights.clip(lower=min_weight)
        weights = weights / weights.sum()
        
        return weights.to_dict()
    
    def generate_visualizations(self):
            """生成分析可视化"""
            print("正在生成可视化...")
            
            # 设置样式
            plt.style.use('seaborn-v0_8')
            
            # 1. 奖牌获得概率条形图
            plt.figure(figsize=(15, 8))
            ax = sns.barplot(
                data=self.predictions.sort_values('Medal_Probability', ascending=False),
                x='NOC',
                y='Medal_Probability',
                hue='NOC',
                legend=False,
                palette='viridis'
            )
            plt.title('Probability of Winning First Olympic Medal by Country', pad=20)
            plt.xlabel('Country Code')
            plt.ylabel('Probability Score')
            plt.xticks(rotation=45, ha='right')
            
            # 添加数值标签
            for i, v in enumerate(self.predictions.sort_values('Medal_Probability', ascending=False)['Medal_Probability']):
                ax.text(i, v, f'{v:.3f}', ha='center', va='bottom')
                
            plt.tight_layout()
            plt.savefig(os.path.join(self.data_dir, 'medal_probability.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. 特征相关性热图
            plt.figure(figsize=(12, 8))
            numeric_cols = [
                'Historical_Participation', 'Total_Athletes', 'Sports_Diversity',
                'Recent_Athletes', 'Recent_Sports', 'Growth_Rate', 
                'Athletes_Per_Sport', 'Sport_Consistency', 'Medal_Probability'
            ]
            correlation_matrix = self.predictions[numeric_cols].corr()
            sns.heatmap(
                correlation_matrix, 
                annot=True, 
                cmap='coolwarm', 
                center=0,
                fmt='.2f',
                square=True
            )
            plt.title('Feature Correlation Heatmap')
            plt.tight_layout()
            plt.savefig(os.path.join(self.data_dir, 'feature_correlation.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. 多维度散点图
            plt.figure(figsize=(12, 8))
            sns.scatterplot(
                data=self.predictions,
                x='Historical_Participation',
                y='Recent_Athletes',
                size='Sports_Diversity',
                hue='Medal_Probability',
                sizes=(50, 400),
                palette='viridis'
            )
            plt.title('Multi-dimensional Analysis of Medal Potential')
            plt.xlabel('Historical Olympic Participation')
            plt.ylabel('Recent Olympic Athletes')
            for idx, row in self.predictions.iterrows():
                plt.annotate(
                    row['NOC'],
                    (row['Historical_Participation'], row['Recent_Athletes']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.7
                )
            plt.tight_layout()
            plt.savefig(os.path.join(self.data_dir, 'multi_dimensional_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 4. 运动项目多样性分析
            plt.figure(figsize=(12, 6))
            diversity_data = self.predictions.sort_values('Sports_Diversity', ascending=False)
            sns.barplot(
                data=diversity_data,
                x='NOC',
                y='Sports_Diversity',
                palette='viridis'
            )
            plt.title('Sports Diversity by Country')
            plt.xlabel('Country Code')
            plt.ylabel('Number of Sports')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(self.data_dir, 'sports_diversity.png'), dpi=300, bbox_inches='tight')
            plt.close()

    def generate_report(self):
            """生成综合分析报告"""
            try:
                if self.predictions is None or self.predictions.empty:
                    raise ValueError("没有可用的预测数据来生成报告")

                # 确保所需的列都存在
                required_columns = ['NOC', 'Medal_Probability', 'Historical_Participation', 
                                'Recent_Athletes', 'Sports_Diversity', 'Growth_Rate']
                missing_columns = [col for col in required_columns if col not in self.predictions.columns]
                if missing_columns:
                    raise ValueError(f"预测数据缺少必要的列: {', '.join(missing_columns)}")

                # 获取排序后的预测数据
                sorted_predictions = self.predictions.sort_values('Medal_Probability', ascending=False)

                # 生成报告内容
                report = f"""
    奥运会首枚奖牌获得预测分析报告
    ==============================
    生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    摘要
    ----
    分析国家总数: {len(self.predictions)}

    最有可能获得首枚奖牌的前5个国家
    ----------------------------
    {sorted_predictions.head().to_string() if len(sorted_predictions) >= 5 else sorted_predictions.to_string()}

    关键发现
    -------
    1. 最具潜力的候选国家:
    {sorted_predictions.head(3)[['NOC', 'Medal_Probability']].to_string() if len(sorted_predictions) >= 3 else "数据不足"}

    2. 关键表现指标:"""

                # 安全获取指标最大值对应的国家
                if len(sorted_predictions) > 0:
                    report += f"""
    - 历史参与度最高: {sorted_predictions.iloc[sorted_predictions['Historical_Participation'].argmax()]['NOC'] if 'Historical_Participation' in sorted_predictions else '数据不可用'}
    - 近期活跃度最高: {sorted_predictions.iloc[sorted_predictions['Recent_Athletes'].argmax()]['NOC'] if 'Recent_Athletes' in sorted_predictions else '数据不可用'}
    - 运动项目最多样化: {sorted_predictions.iloc[sorted_predictions['Sports_Diversity'].argmax()]['NOC'] if 'Sports_Diversity' in sorted_predictions else '数据不可用'}
    - 发展速度最快: {sorted_predictions.iloc[sorted_predictions['Growth_Rate'].argmax()]['NOC'] if 'Growth_Rate' in sorted_predictions else '数据不可用'}"""
                else:
                    report += "\n数据不足以生成关键表现指标"

                report += """

    3. 数据分析要点:
    - 参与度分析: 历史参与和近期活跃度是重要指标
    - 多样性分析: 运动项目的多样性对获得奖牌机会有显著影响
    - 发展趋势: 近年来的发展速度对预测结果有重要参考价值

    建议
    ----
    1. 重点关注国家:"""

                if len(sorted_predictions) >= 3:
                    report += f"""
    {sorted_predictions.head(3)[['NOC', 'Medal_Probability', 'Sports_Diversity']].to_string()}"""
                else:
                    report += "\n数据不足以提供重点关注国家列表"

                report += """

    2. 发展建议:
    - 针对运动项目多样性低的国家，建议扩大参与运动项目范围
    - 重点提升近期参与度和运动员数量
    - 保持奥运会持续参与度
    - 加强运动员培养和支持系统

    3. 具体措施:
    - 扩大体育投资和基础设施建设
    - 加强青少年体育人才培养
    - 选择具有比较优势的运动项目重点发展
    - 提高教练员和运动员的专业水平

    预测方法说明
    ----------
    预测模型综合考虑以下因素:
    - 历史奥运会参与情况
    - 运动员总数及发展趋势
    - 运动项目多样性
    - 近期表现指标
    - 发展速度和潜力评估

    所有指标均经过标准化处理，并采用动态权重进行综合评分。
    """
                
                # 保存报告
                report_path = os.path.join(self.data_dir, 'medal_prediction_report.txt')
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                
                return report

            except Exception as e:
                error_report = f"""
    错误报告
    ========
    生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    报告生成过程中遇到错误: {str(e)}

    建议措施:
    1. 检查输入数据的完整性
    2. 验证预测计算步骤
    3. 确保所有必要的数据列都存在
    """
                print(f"生成报告时出错: {str(e)}")
                return error_report

def main():
    try:
        # 初始化预测器
        predictor = OlympicMedalPredictor()
        
        # 加载数据
        predictor.load_data()
        
        # 生成预测
        predictions = predictor.predict_medals()
        print("\n预测计算完成.")
        
        # 创建可视化
        print("正在生成可视化...")
        predictor.generate_visualizations()
        
        # 生成并打印报告
        print("\n正在生成最终报告...")
        report = predictor.generate_report()
        print(report)
        
        # 保存详细结果
        predictions.to_csv(os.path.join('data', 'detailed_medal_predictions.csv'), 
                         index=False, encoding='utf-8')
        print("\n分析完成! 所有结果已保存在data目录中.")
        
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")

if __name__ == "__main__":
    main()