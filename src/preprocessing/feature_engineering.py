"""Feature engineering module."""
import pandas as pd
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates domain-specific features for loan default prediction."""
    
    def __init__(self):
        """Initialize feature engineer."""
        logger.info("Initialized FeatureEngineer")
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all engineered features.
        
        Args:
            df: DataFrame to engineer features for
            
        Returns:
            DataFrame with new features
        """
        df = df.copy()
        logger.info("Starting feature engineering")
        
        # Age features
        if 'Age_Days' in df.columns:
            df['Age_Years'] = abs(df['Age_Days']) / 365.25
        
        # Employment features
        if 'Employed_Days' in df.columns:
            df['Employment_Years'] = abs(df['Employed_Days']) / 365.25
            df['Is_Unemployed'] = (df['Employed_Days'] > 0).astype(int)
        
        # Financial ratios
        if 'Credit_Amount' in df.columns and 'Client_Income' in df.columns:
            df['Credit_Income_Ratio'] = df['Credit_Amount'] / (df['Client_Income'] + 1)
        
        if 'Loan_Annuity' in df.columns and 'Client_Income' in df.columns:
            df['Annuity_Income_Ratio'] = df['Loan_Annuity'] / (df['Client_Income'] + 1)
        
        if 'Loan_Annuity' in df.columns and 'Credit_Amount' in df.columns:
            df['Annuity_Credit_Ratio'] = df['Loan_Annuity'] / (df['Credit_Amount'] + 1)
        
        # Asset features
        asset_cols = ['Car_Owned', 'Bike_Owned', 'House_Own']
        if all(col in df.columns for col in asset_cols):
            df['Total_Assets'] = df[asset_cols].sum(axis=1)
        
        if 'Active_Loan' in df.columns and 'Total_Assets' in df.columns:
            df['Financial_Stability'] = df['Total_Assets'] - df['Active_Loan']
        
        # Credit bureau features
        if 'Credit_Bureau' in df.columns:
            df['High_Bureau_Activity'] = (df['Credit_Bureau'] > 3).astype(int)
        
        # Social circle features
        if 'Social_Circle_Default' in df.columns:
            df['Social_Risk_Flag'] = (df['Social_Circle_Default'] > 0).astype(int)
        
        # External score features
        score_cols = ['Score_Source_1', 'Score_Source_2', 'Score_Source_3']
        if all(col in df.columns for col in score_cols):
            df['Avg_External_Score'] = df[score_cols].mean(axis=1)
            df['Min_External_Score'] = df[score_cols].min(axis=1)
        
        # Contact features
        contact_cols = ['Mobile_Tag', 'Homephone_Tag', 'Workphone_Working']
        if all(col in df.columns for col in contact_cols):
            df['Contact_Completeness'] = df[contact_cols].sum(axis=1)
        
        # Document change features
        doc_cols = ['Registration_Days', 'ID_Days', 'Phone_Change']
        if all(col in df.columns for col in doc_cols):
            df['Recent_Doc_Changes'] = (df[doc_cols] < 365).sum(axis=1)
        
        # Age groups
        if 'Age_Years' in df.columns:
            age_group = pd.cut(
                df['Age_Years'],
                bins=[0, 25, 35, 45, 55, 100],
                labels=[0, 1, 2, 3, 4]
            )
            df['Age_Group'] = age_group.cat.codes.replace(-1, np.nan).fillna(0).astype(int)
        
        # Income levels
        if 'Client_Income' in df.columns:
            try:
                income_level = pd.qcut(
                    df['Client_Income'],
                    q=5,
                    labels=[0, 1, 2, 3, 4],
                    duplicates='drop'
                )
                df['Income_Level'] = income_level.cat.codes.replace(-1, np.nan).fillna(0).astype(int)
            except ValueError:
                logger.warning("Could not create Income_Level due to insufficient unique values")
        
        # Family size categories
        if 'Client_Family_Members' in df.columns:
            family_cat = pd.cut(
                df['Client_Family_Members'],
                bins=[-1, 2, 4, 10],
                labels=[0, 1, 2]
            )
            df['Family_Size_Category'] = family_cat.cat.codes.replace(-1, np.nan).fillna(0).astype(int)
        
        # House age categories
        if 'Own_House_Age' in df.columns:
            house_cat = pd.cut(
                df['Own_House_Age'],
                bins=[-1, 10, 20, 50, 100],
                labels=[0, 1, 2, 3]
            )
            df['House_Age_Category'] = house_cat.cat.codes.replace(-1, np.nan).fillna(0).astype(int)
        
        # Application timing features
        if 'Application_Process_Hour' in df.columns:
            df['Business_Hours'] = (
                (df['Application_Process_Hour'] >= 9) &
                (df['Application_Process_Hour'] <= 17)
            ).astype(int)
        
        if 'Application_Process_Day' in df.columns:
            df['Weekend_Application'] = (df['Application_Process_Day'].isin([0, 6])).astype(int)
        
        # Address mismatch
        mismatch_cols = ['Client_Permanent_Match_Tag', 'Client_Contact_Work_Tag']
        if all(col in df.columns for col in mismatch_cols):
            df['Address_Mismatch_Score'] = df[mismatch_cols].sum(axis=1)
        
        # Dependents ratio
        if 'Child_Count' in df.columns and 'Client_Income' in df.columns:
            df['Dependents_Income_Ratio'] = df['Child_Count'] / (df['Client_Income'] / 10000 + 1)
        
        # Credit history
        if 'Registration_Days' in df.columns:
            df['Credit_History_Years'] = abs(df['Registration_Days']) / 365.25
        
        logger.info(f"Feature engineering completed. Total features: {df.shape[1]}")
        
        return df
    
    def get_feature_names(self, df: pd.DataFrame, exclude_cols: List[str] = None) -> List[str]:
        """
        Get list of feature names after engineering.
        
        Args:
            df: DataFrame with engineered features
            exclude_cols: Columns to exclude from feature list
            
        Returns:
            List of feature names
        """
        if exclude_cols is None:
            exclude_cols = []
        
        feature_names = [col for col in df.columns if col not in exclude_cols]
        return feature_names

