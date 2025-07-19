#!/usr/bin/env python3
"""
Healthcare Appointment No-Show Prediction Feature Engineering Pipeline

This script processes appointment data for no-show prediction with complete
feature engineering. Designed for standalone CSV processing with no external
dependencies on proprietary systems.

Usage:
    python preprocessing.py input_file.csv [output_file.csv]
"""

import pandas as pd
import numpy as np
import sys
import warnings
from typing import Dict, List, Optional

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configuration constants
CONFIG = {
    'STATUS_MAPPING': {
        'Invoiced': 'Show',
        'Visited': 'No Show',
        'Confirmed': 'No Show',
        'Cancelled': 'No Show',
        'Canceled': 'No Show',
        'Not Answered': 'No Show',
        'Booked': 'No Show',
    },
    'AGE_BINS': [0, 1, 5, 12, 18, 30, 45, 60, 75, 120],
    'AGE_LABELS': ['<1y', '1-4y', '5-11y', '12-17y', '18-29y', '30-44y', '45-59y', '60-74y', '75+'],
    'RECENCY_BINS': [-np.inf, 30, 90, 180, 365, np.inf],
    'RECENCY_LABELS': ['≤30d', '31-90d', '91-180d', '181-365d', '>365d'],
    'LEADTIME_BINS': [-np.inf, 1, 7, 30, 90, np.inf],
    'LEADTIME_LABELS': ['<1d', '1-7d', '8-30d', '31-90d', '>90d'],
    'COLUMNS_TO_DROP': ['PaymentMode', 'VisitType', 'doctor_Nationality', 'District', 'CustomeNumber', 'Job_Location', 'Occupation', 'company'],
    'REQUIRED_COLUMNS': ['BranchCode', 'DOB', 'Location', 'AppointmentDate', 'Status', 'DoctorName', 'Department'],
}


class HealthcarePreprocessor:
    """
    A preprocessing pipeline for healthcare no-show prediction data.
    Processes CSV files with feature engineering for ML models.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize the preprocessor with configuration."""
        self.config = config or CONFIG
        
    def load_data_from_csv(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            print(f"Loading data from CSV file: {file_path}")
            df = pd.read_csv(file_path)
            print(f"Successfully loaded {len(df)} rows from CSV")
            return df
        except Exception as e:
            print(f"Error loading data from CSV: {e}")
            raise
    
    def validate_input_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate input data schema and quality."""
        print(f"Validating input data: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Check required columns
        missing_cols = [col for col in self.config['REQUIRED_COLUMNS'] if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if df.empty:
            raise ValueError("Input dataframe is empty")
        
        print(f"Data validation completed")
        return df

    def clean_initial_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Initial data cleaning and preparation."""
        print("Starting initial data cleaning")
        
        df = df.copy()
        initial_rows = len(df)
        
        # Filter for only No Show records
        if 'Status' in df.columns:
            no_show_statuses = ['Confirmed','Not Answered', 'Booked', 'Visited']
            df = df[df['Status'].isin(no_show_statuses)]
            print(f"Filtered for records: {initial_rows} -> {len(df)} rows")
        
        # Drop specified columns safely
        cols_to_drop = [col for col in self.config['COLUMNS_TO_DROP'] if col in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
        
        # Fill missing values
        if 'LastAppointmentStatus' in df.columns:
            df['LastAppointmentStatus'] = df['LastAppointmentStatus'].fillna('No Prior Visit')
        if 'Previous_Payment_Mode' in df.columns:
            df['Previous_Payment_Mode'] = df['Previous_Payment_Mode'].fillna('FirstTime')
        
        print("Initial data cleaning completed")
        return df

    def process_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process age-related features."""
        print("Processing age features")
        
        # Parse dates safely
        df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')
        df['AppointmentDate'] = pd.to_datetime(df['AppointmentDate'], errors='coerce')
        
        # Calculate age features
        days_alive = (df['AppointmentDate'] - df['DOB']).dt.days
        df['age_at_visit'] = (days_alive // 365).astype('Int64')
        
        # Birth date features
        df['birth_year'] = df['DOB'].dt.year.astype('Int64')
        df['birth_month'] = df['DOB'].dt.month.astype('Int64')
        df['birth_day'] = df['DOB'].dt.day.astype('Int64')
        df['birth_dayofweek'] = df['DOB'].dt.dayofweek.astype('Int64')
        
        # Age bands
        df['age_band'] = pd.cut(
            df['age_at_visit'],
            bins=self.config['AGE_BINS'],
            labels=self.config['AGE_LABELS'],
            right=False
        )
        
        # Clean invalid ages
        df = df.dropna(subset=['age_at_visit'])
        
        return df

    def process_appointment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process appointment date and time features."""
        print("Processing appointment features")
        
        # Ensure appointment date is parsed
        df['AppointmentDate'] = pd.to_datetime(df['AppointmentDate'], errors='coerce')
        
        # Calendar features
        df['appt_year'] = df['AppointmentDate'].dt.year.astype('Int64')
        df['appt_month'] = df['AppointmentDate'].dt.month.astype('Int64')
        df['appt_day'] = df['AppointmentDate'].dt.day.astype('Int64')
        df['appt_quarter'] = df['AppointmentDate'].dt.quarter.astype('Int64')
        df['appt_weekofyear'] = df['AppointmentDate'].dt.isocalendar().week.astype('Int64')
        df['appt_dayofweek'] = df['AppointmentDate'].dt.dayofweek.astype('Int64')
        
        # Weekend flag (Friday, Saturday)
        df['is_weekend'] = df['appt_dayofweek'].isin([4, 5]).astype('Int8')
        
        # Week of month
        df['appt_weekofmonth'] = ((df['AppointmentDate'].dt.day - 1) // 7 + 1).astype('Int64')
        
        # Seasonal features
        df['season'] = df['appt_month'].apply(self._get_season).astype('category')
        
        return df
    
    def _get_season(self, m):
        """Determine season from month."""
        if pd.isna(m):
            return 'unknown'
        if m in [5, 6, 7, 8, 9]:
            return 'hot'
        elif m in [10, 11, 3, 4]:
            return 'warm'
        else:
            return 'mild'

    def process_billing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process billing-related features."""
        print("Processing billing features")
        
        if 'Previous_Bill_Date' in df.columns:
            # Parse billing date
            df['Previous_Bill_Date'] = pd.to_datetime(df['Previous_Bill_Date'], errors='coerce')
            
            # Time since last bill
            days_diff = (df['AppointmentDate'] - df['Previous_Bill_Date']).dt.days
            df['days_since_prev_bill'] = days_diff
            
            # Recency buckets
            df['recency_bucket'] = pd.cut(
                days_diff,
                bins=self.config['RECENCY_BINS'],
                labels=self.config['RECENCY_LABELS']
            )
        
        return df

    def process_booking_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process booking-related features."""
        print("Processing booking features")
        
        if 'Booked_Date_Time' in df.columns:
            # Parse booking datetime
            df['Booked_Date_Time'] = pd.to_datetime(df['Booked_Date_Time'], errors='coerce')
            
            # Calendar features
            df['book_year'] = df['Booked_Date_Time'].dt.year.astype('Int64')
            df['book_month'] = df['Booked_Date_Time'].dt.month.astype('Int64')
            df['book_dayofweek'] = df['Booked_Date_Time'].dt.dayofweek.astype('Int64')
            df['book_hour'] = df['Booked_Date_Time'].dt.hour.astype('Int64')
            
            # Lead time features
            lead_timedelta = df['AppointmentDate'] - df['Booked_Date_Time']
            lead_days = (lead_timedelta.dt.total_seconds() / 86400).round(1)
            
            # Lead time buckets
            df['leadtime_bucket'] = pd.cut(
                lead_days,
                bins=self.config['LEADTIME_BINS'],
                labels=self.config['LEADTIME_LABELS']
            )
            
            # Same day booking
            df['same_day_booking'] = (lead_days <= 0.0).astype('Int8')
        
        return df

    def process_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process categorical features."""
        print("Processing categorical features")
        
        # Group rare categories
        if 'Nationality' in df.columns:
            df['Nationality_grouped'] = self._group_rare(df['Nationality'], top_n=10)
        
        if 'Location' in df.columns:
            df['Location_cleaned'] = df['Location'].str.lower().str.strip()
            df['Location_grouped'] = self._group_rare(df['Location_cleaned'], top_n=15)
        
        return df
    
    def _group_rare(self, series: pd.Series, top_n: int = 10) -> pd.Series:
        """Group rare categories into 'Other'."""
        try:
            top_values = series.value_counts().nlargest(top_n).index
            return series.apply(lambda x: x if x in top_values else 'Other')
        except:
            return series

    def process_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process target variable."""
        print("Processing target variable")
        
        # Since we're filtering for only No Show records, all targets are 'No Show'
        df['Target'] = 'No Show'
        
        return df

    def final_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final cleanup and data type optimization."""
        print("Performing final cleanup")
        
        # Drop intermediate columns
        columns_to_drop = ['Status']
        existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
        
        if existing_cols_to_drop:
            df = df.drop(columns=existing_cols_to_drop)
        
        # Convert object columns to category for memory efficiency
        object_cols = df.select_dtypes(include=['object']).columns
        for col in object_cols:
            df[col] = df[col].astype('category')
        
        return df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main preprocessing pipeline."""
        print("Starting preprocessing pipeline")
        
        # Validate input
        df = self.validate_input_data(df)
        
        # Process all feature groups
        df = self.clean_initial_data(df)
        df = self.process_age_features(df)
        df = self.process_appointment_features(df)
        df = self.process_billing_features(df)
        df = self.process_booking_features(df)
        df = self.process_categorical_features(df)
        df = self.process_target_variable(df)
        df = self.final_cleanup(df)
        
        print(f"Preprocessing pipeline completed. Final shape: {df.shape}")
        return df


def main():
    """Main execution function for CSV processing."""
    if len(sys.argv) < 2:
        print("Usage: python preprocessing.py <input_file.csv> [output_file.csv]")
        sys.exit(1)
    
    try:
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else f"processed_{input_path}"
        
        # Initialize preprocessor
        preprocessor = HealthcarePreprocessor()
        
        print(f"Processing: {input_path} -> {output_path}")
        
        # Load data from CSV
        df = preprocessor.load_data_from_csv(input_path)
        initial_shape = df.shape
        
        # Process data
        df_processed = preprocessor.preprocess_data(df)
        
        # Save to CSV
        df_processed.to_csv(output_path, index=False)
        
        print(f"✅ Feature engineering complete!")
        print(f"   Input: {initial_shape[0]:,} rows × {initial_shape[1]} columns")
        print(f"   Output: {df_processed.shape[0]:,} rows × {df_processed.shape[1]} columns")
        print(f"   File saved: {output_path}")
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise


if __name__ == "__main__":
    main()
