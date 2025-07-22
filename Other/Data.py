import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def encode_data_for_ml(df: pd.DataFrame, encoding_type: str = 'label') -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df_encoded = df.copy()
    categorical_columns = []
    numerical_columns = []
    
    for col in df.columns:
        if df[col].dtype in ['object', 'category']:
            categorical_columns.append(col)
        elif df[col].dtype in ['bool']:
            categorical_columns.append(col)
        elif df[col].nunique() <= 10 and df[col].dtype in ['int64', 'int32']:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.05:  
                categorical_columns.append(col)
            else:
                numerical_columns.append(col)
        else:
            numerical_columns.append(col)
    
    logger.info(f"Detected {len(categorical_columns)} categorical columns: {categorical_columns}")
    logger.info(f"Detected {len(numerical_columns)} numerical columns")
    encoding_metadata = {
        'encoding_type': encoding_type,
        'categorical_columns': categorical_columns,
        'numerical_columns': numerical_columns,
        'encoders': {},
        'new_columns': [],
        'dropped_columns': []
    }
    
    if not categorical_columns:
        logger.info("No categorical columns found. Returning original DataFrame.")
        return df_encoded, encoding_metadata
    
    if encoding_type == 'label':
        logger.info("Applying Label Encoding to categorical columns...")
        
        for col in categorical_columns:
            original_col = df_encoded[col].copy()
            df_encoded[col] = df_encoded[col].fillna('_MISSING_')
            le = LabelEncoder()
            encoded_values = le.fit_transform(df_encoded[col].astype(str))
            df_encoded[col] = pd.Series(encoded_values, index=df_encoded.index, dtype='float64')
            encoding_metadata['encoders'][col] = {
                'encoder': le,
                'classes': le.classes_.tolist(),
                'mapping': dict(zip(le.classes_, le.transform(le.classes_)))
            }
            
            logger.info(f"Label encoded column '{col}': {len(le.classes_)} unique values")
    
    elif encoding_type == 'onehot':
        logger.info("Applying One-Hot Encoding to categorical columns...")
        feasibility_check = check_encoding_feasibility(df)
        if not feasibility_check['is_safe_for_onehot']:
            logger.error("⚠️  One-hot encoding not recommended due to high cardinality columns!")
            for warning in feasibility_check['warnings']:
                logger.error(warning)
            encoding_metadata['feasibility_warnings'] = feasibility_check
        
        for col in categorical_columns:
            unique_count = df[col].nunique()
            if unique_count > 100:
                logger.error(f"SKIPPING column '{col}': {unique_count} unique values would create too many columns!")
                encoding_metadata['skipped_columns'] = encoding_metadata.get('skipped_columns', [])
                encoding_metadata['skipped_columns'].append({
                    'column': col,
                    'reason': f'Too many unique values ({unique_count})',
                    'recommendation': 'Use label encoding instead'
                })
                continue  
            df_encoded[col] = df_encoded[col].fillna('_MISSING_')
            dummies = pd.get_dummies(df_encoded[col], prefix=col, dummy_na=False, dtype='float64')
            original_values = df_encoded[col].unique().tolist()
            df_encoded = df_encoded.drop(columns=[col])
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            encoding_metadata['dropped_columns'].append(col)
            encoding_metadata['new_columns'].extend(dummies.columns.tolist())
            encoding_metadata['encoders'][col] = {
                'original_values': original_values,
                'dummy_columns': dummies.columns.tolist(),
                'type': 'onehot'
            }
            
            logger.info(f"One-hot encoded column '{col}': created {len(dummies.columns)} dummy columns")
    
    else:
        raise ValueError(f"Invalid encoding_type: {encoding_type}. Use 'label' or 'onehot'.")
    
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            logger.warning(f"Column '{col}' still contains object data after encoding. Converting to numeric.")
            df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce').fillna(0.0)
    
    logger.info(f"Encoding complete. DataFrame shape: {df.shape} -> {df_encoded.shape}")
    
    return df_encoded, encoding_metadata


def get_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    
    categorical_columns = []
    numerical_columns = []
    
    for col in df.columns:
        if df[col].dtype in ['object', 'category']:
            categorical_columns.append(col)
        elif df[col].dtype in ['bool']:
            categorical_columns.append(col)
        elif df[col].nunique() <= 10 and df[col].dtype in ['int64', 'int32']:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.05:  
                categorical_columns.append(col)
            else:
                numerical_columns.append(col)
        else:
            numerical_columns.append(col)
    
    return categorical_columns, numerical_columns
    
def prepare_visualization_data(df: pd.DataFrame) -> Dict[str, Any]:
    
    categorical_columns, numerical_columns = get_column_types(df)
    
    viz_data = {
        'correlation_matrix': {},
        'distribution_data': {},
        'categorical_distributions': {},
        'missing_data_heatmap': {},
        'basic_stats': {}
    }
    if len(numerical_columns) > 1:
        corr_matrix = df[numerical_columns].corr()
        viz_data['correlation_matrix'] = {
            'data': corr_matrix.values.tolist(),
            'columns': corr_matrix.columns.tolist(),
            'index': corr_matrix.index.tolist()
        }
    for col in numerical_columns:
        col_data = df[col].dropna()
        if len(col_data) > 0:
            viz_data['distribution_data'][col] = {
                'values': col_data.tolist(),
                'bins': np.histogram(col_data, bins=30)[1].tolist(),
                'counts': np.histogram(col_data, bins=30)[0].tolist(),
                'quartiles': col_data.quantile([0.25, 0.5, 0.75]).tolist(),
                'mean': col_data.mean(),
                'std': col_data.std()
            }
    for col in categorical_columns:
        value_counts = df[col].value_counts()
        viz_data['categorical_distributions'][col] = {
            'categories': value_counts.index.tolist(),
            'counts': value_counts.values.tolist(),
            'percentages': (value_counts / len(df) * 100).values.tolist()
        }
    missing_data = df.isnull()
    viz_data['missing_data_heatmap'] = {
        'data': missing_data.astype(int).values.tolist(),
        'columns': missing_data.columns.tolist(),
        'missing_counts': missing_data.sum().tolist(),
        'missing_percentages': (missing_data.sum() / len(df) * 100).tolist()
    }
    viz_data['basic_stats'] = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'categorical_columns': len(categorical_columns),
        'numerical_columns': len(numerical_columns),
        'missing_values_total': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    return viz_data


def analyze_csv_data(df: pd.DataFrame) -> Dict[str, Any]:
    
    categorical_columns, numerical_columns = get_column_types(df)
    encoding_check = check_encoding_feasibility(df)
    
    results = {
        'column_types': {
            'categorical': categorical_columns,
            'numerical': numerical_columns
        },
        'basic_info': {
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'dtypes': {col: str(df[col].dtype) for col in df.columns}
        },
        'encoding_feasibility': encoding_check,   
        'visualization_data': prepare_visualization_data(df)
    }
    if not encoding_check['is_safe_for_onehot']:
        logger.warning("⚠️  One-hot encoding may cause memory issues. Check 'encoding_feasibility' in results.")
    
    return results


def prepare_data_for_ml(df: pd.DataFrame, target_column: str, 
                       encoding_method: str = 'label') -> Tuple[pd.DataFrame, Dict[str, Any]]:
    
    categorical_columns, numerical_columns = get_column_types(df)
    df_processed, encoding_info = encode_data_for_ml(df, encoding_type=encoding_method)
    
    for col in numerical_columns:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    preprocessing_info = {
        'encoding_info': encoding_info,
        'original_shape': df.shape,
        'final_shape': df_processed.shape,
        'categorical_columns': categorical_columns,
        'numerical_columns': numerical_columns
    }
    
    return df_processed, preprocessing_info

def check_encoding_feasibility(df: pd.DataFrame) -> Dict[str, Any]:
    
    categorical_columns, numerical_columns = get_column_types(df)
    
    warnings = []
    recommendations = []
    high_cardinality_columns = []
    estimated_memory_increase = 1.0  
    
    for col in categorical_columns:
        unique_count = df[col].nunique()
        if unique_count > 100:
            high_cardinality_columns.append({
                'column': col,
                'unique_values': unique_count,
                'severity': 'critical'
            })
            warnings.append(f"CRITICAL: Column '{col}' has {unique_count} unique values. One-hot encoding will create {unique_count} new columns!")
            recommendations.append(f"Use label encoding for '{col}' instead of one-hot encoding")
            estimated_memory_increase *= unique_count
            
        elif unique_count > 50:
            high_cardinality_columns.append({
                'column': col,
                'unique_values': unique_count,
                'severity': 'warning'
            })
            warnings.append(f"WARNING: Column '{col}' has {unique_count} unique values. This will significantly increase memory usage.")
            recommendations.append(f"Consider label encoding for '{col}' to reduce memory usage")
            estimated_memory_increase *= unique_count * 0.5
            
        elif unique_count > 20:
            high_cardinality_columns.append({
                'column': col,
                'unique_values': unique_count,
                'severity': 'info'
            })
            warnings.append(f"INFO: Column '{col}' has {unique_count} unique values. One-hot encoding is feasible but will add {unique_count} columns.")
            estimated_memory_increase *= unique_count * 0.2
    original_columns = len(df.columns)
    total_new_columns = sum(col_info['unique_values'] for col_info in high_cardinality_columns)
    estimated_final_columns = original_columns + total_new_columns - len(categorical_columns)
    current_memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    estimated_memory_mb = current_memory_mb * estimated_memory_increase
    overall_recommendation = "safe"
    if any(col['severity'] == 'critical' for col in high_cardinality_columns):
        overall_recommendation = "avoid_onehot"
    elif any(col['severity'] == 'warning' for col in high_cardinality_columns):
        overall_recommendation = "use_with_caution"
    
    result = {
        'is_safe_for_onehot': overall_recommendation == "safe",
        'overall_recommendation': overall_recommendation,
        'warnings': warnings,
        'recommendations': recommendations,
        'high_cardinality_columns': high_cardinality_columns,
        'memory_estimate': {
            'current_memory_mb': round(current_memory_mb, 2),
            'estimated_memory_mb': round(estimated_memory_mb, 2),
            'memory_increase_factor': round(estimated_memory_increase, 2)
        },
        'column_estimate': {
            'current_columns': original_columns,
            'estimated_final_columns': estimated_final_columns,
            'new_columns_added': total_new_columns
        },
        'categorical_summary': {
            'total_categorical': len(categorical_columns),
            'safe_for_onehot': len([col for col in categorical_columns if df[col].nunique() <= 20]),
            'risky_for_onehot': len([col for col in categorical_columns if df[col].nunique() > 20])
        }
    }
    
    logger.info(f"Encoding feasibility check: {overall_recommendation}")
    if warnings:
        for warning in warnings[:3]:  
            logger.warning(warning)
    
    return result

def map_original_to_encoded_columns(original_columns: List[str], encoding_metadata: Dict[str, Any], 
                                  df_encoded: pd.DataFrame) -> List[str]:
    mapped_columns = []
    
    for col in original_columns:
        if col in df_encoded.columns:
            mapped_columns.append(col)
        elif col in encoding_metadata.get('encoders', {}):
            encoder_info = encoding_metadata['encoders'][col]
            if encoder_info.get('type') == 'onehot':
                dummy_columns = encoder_info.get('dummy_columns', [])
                mapped_columns.extend(dummy_columns)
            else:
                mapped_columns.append(col)
        else:
            logger.warning(f"Column '{col}' not found in encoded DataFrame. Skipping.")
    
    return mapped_columns


