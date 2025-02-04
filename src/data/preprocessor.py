import pandas as pd
import numpy as np
from rich.progress import track
from rich.console import Console
from typing import Dict, List, Optional
from datetime import datetime

class OlympicDataPreprocessor:
    def __init__(self):
        self.console = Console()
        
    def preprocess_athletes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess athletes dataset"""
        with self.console.status("[bold green]Preprocessing athletes data...") as status:
            df = df.copy()
            
            # Clean names
            df['Name'] = df['Name'].str.strip()
            
            # Create age category for athletes (if birth year is available)
            if 'Age' in df.columns:
                df['Age_Category'] = pd.cut(df['Age'], 
                                          bins=[0, 20, 25, 30, 35, 100],
                                          labels=['Under 20', '20-25', '26-30', '31-35', 'Over 35'])
            
            # Create medal indicator
            df['Has_Medal'] = df['Medal'].notna().astype(int)
            
            # Create participation count
            participation_count = df.groupby('Name').Year.count().reset_index()
            participation_count.columns = ['Name', 'Participation_Count']
            df = df.merge(participation_count, on='Name', how='left')
            
            self.console.log("Athletes preprocessing completed")
            return df
            
    def preprocess_medal_counts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess medal counts dataset"""
        with self.console.status("[bold green]Preprocessing medal counts...") as status:
            df = df.copy()
            
            # Calculate additional metrics
            df['Gold_Ratio'] = df['Gold'] / df['Total']
            df['Medal_Points'] = df['Gold']*3 + df['Silver']*2 + df['Bronze']
            
            # Create medal efficiency (medals per event - needs programs data)
            # This would be added later when combining datasets
            
            # Sort values
            df = df.sort_values(['Year', 'Total', 'Gold'], ascending=[True, False, False])
            
            self.console.log("Medal counts preprocessing completed")
            return df

    def preprocess_hosts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess hosts dataset"""
        with self.console.status("[bold green]Preprocessing hosts data...") as status:
            df = df.copy()
            
            # Extract city and country
            df[['City', 'Country']] = df['Host'].str.extract(r'([^,]+),\s*(.+)')
            
            # Clean cancelled events
            df['Cancelled'] = df['Host'].str.contains('Cancelled', case=False).fillna(False)
            
            # Create time since last hosted
            df['Years_Since_Last'] = df['Year'].diff()
            
            self.console.log("Hosts preprocessing completed")
            return df

    def preprocess_programs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess programs dataset"""
        with self.console.status("[bold green]Preprocessing programs data...") as status:
            df = df.copy()
            
            # Convert bullet points to 0
            df = df.replace('•', 0)
            
            # Convert to numeric where possible
            numeric_cols = df.columns[df.columns.str.match(r'^\d{4}$')]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Calculate total events per sport
            df['Total_Events'] = df[numeric_cols].sum(axis=1)
            
            self.console.log("Programs preprocessing completed")
            return df

    def create_combined_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create combined features from multiple datasets"""
        with self.console.status("[bold green]Creating combined features...") as status:
            # Combine medal counts with host information
            combined = data['medal_counts'].merge(
                data['hosts'][['Year', 'Host', 'City', 'Country']], 
                on='Year', 
                how='left'
            )
            
            # Add host country indicator
            combined['Is_Host'] = (combined['NOC'] == combined['Country']).astype(int)
            
            # Add total events that year (from programs)
            yearly_events = data['programs'][data['programs'].columns[
                data['programs'].columns.str.match(r'^\d{4}$')]].sum()
            combined = combined.merge(
                pd.DataFrame({'Year': yearly_events.index.astype(int), 
                            'Total_Events': yearly_events.values}),
                on='Year',
                how='left'
            )
            
            # Calculate efficiency metrics
            combined['Medals_Per_Event'] = combined['Total'] / combined['Total_Events']
            
            self.console.log("Combined features created")
            return combined

    def get_feature_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic statistics for numerical features"""
        stats = df.describe()
        stats.loc['missing'] = df.isnull().sum()
        stats.loc['unique'] = df.nunique()
        return stats