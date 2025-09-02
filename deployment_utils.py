#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
部署专用工具模块：
- 提供简化的模型加载和预测功能
- 针对 Streamlit 部署优化
- 最小化依赖，最大化兼容性
"""

import pandas as pd
import numpy as np
import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_model_safe(model_path='lead_poisoning_optimized_model.pkl'):
    """安全加载模型文件"""
    try:
        # 首先尝试用joblib加载
        model_data = joblib.load(model_path)
        
        if isinstance(model_data, dict):
            model = model_data.get('model', None)
            if model is None:
                print(f"警告: 模型文件 {model_path} 中没有找到有效的模型对象")
                return None, None
            
            selected_features = model_data.get('selected_features', [])
            return model, {
                'selected_features': selected_features,
                'model_info': f"Model type: {type(model).__name__}",
                'features_count': len(selected_features) if selected_features else 0
            }
        else:
            # 直接是模型对象
            return model_data, {'model_info': f"Model type: {type(model_data).__name__}"}
            
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return None, None

def predict_single_safe(model, features):
    """安全的单个预测"""
    try:
        if not hasattr(model, 'predict_proba'):
            print("错误: 模型没有predict_proba方法")
            return None
        
        # 确保features是正确的格式
        if isinstance(features, list):
            features = np.array(features).reshape(1, -1)
        elif isinstance(features, np.ndarray) and features.ndim == 1:
            features = features.reshape(1, -1)
        
        # 进行预测
        prediction = model.predict_proba(features)[0][1]
        return float(prediction)
        
    except Exception as e:
        print(f"预测失败: {str(e)}")
        return None

def get_risk_interpretation(risk_score):
    """获取风险解释"""
    if risk_score >= 0.8:
        return {
            'level': 'Extremely High Risk',
            'level_cn': '极高风险',
            'color': '#dc3545',
            'suggestions': [
                'Immediate aggressive intervention required',
                'Consider chelation therapy',
                'Remove lead exposure sources',
                'Monitor blood lead levels weekly'
            ]
        }
    elif risk_score >= 0.7:
        return {
            'level': 'High Risk',
            'level_cn': '高风险', 
            'color': '#fd7e14',
            'suggestions': [
                'Close monitoring required',
                'Consider preventive treatment',
                'Monitor blood lead levels bi-weekly',
                'Strengthen environmental intervention'
            ]
        }
    elif risk_score >= 0.4:
        return {
            'level': 'Medium Risk',
            'level_cn': '中等风险',
            'color': '#ffc107',
            'suggestions': [
                'Regular follow-up monitoring',
                'Check blood lead levels monthly',
                'Strengthen health education',
                'Avoid lead contamination sources'
            ]
        }
    else:
        return {
            'level': 'Low Risk',
            'level_cn': '低风险',
            'color': '#28a745',
            'suggestions': [
                'Routine health management',
                'Monitor blood lead levels quarterly',
                'Maintain healthy lifestyle',
                'Regular health checkups'
            ]
        }

def validate_features(features, expected_count=39):
    """验证特征数据"""
    if features is None:
        return False, "特征数据为空"
    
    if isinstance(features, (list, np.ndarray)):
        if len(features) != expected_count:
            return False, f"特征数量不匹配，期望{expected_count}个，实际{len(features)}个"
        
        # 检查是否都是数值
        try:
            np.array(features, dtype=float)
            return True, "特征验证通过"
        except:
            return False, "特征包含非数值数据"
    
    return False, "特征数据格式不正确"

def create_feature_template():
    """创建特征模板"""
    return {
        'Patient_Name': 'Example Patient',
        'Age': 35,
        'Gender': 1,  # 1=Male, 2=Female
        'Blood_Lead_Level': 2.5,
        'Hemoglobin': 130.0,
        'Blood_Calcium': 2.3,
        'Abdominal_Pain': 0,  # 0=No, 1=Yes
        'Total_Bilirubin': 15.0,
        'Detox_Amount': 2.5,
        'Risk_Score': 3.0
    }

def check_deployment_readiness():
    """检查部署就绪状态"""
    issues = []
    
    # 检查模型文件
    try:
        model, model_info = load_model_safe()
        if model is None:
            issues.append("❌ 模型文件无法加载")
        else:
            print("✅ 模型文件加载正常")
    except Exception as e:
        issues.append(f"❌ 模型加载错误: {str(e)}")
    
    # 检查关键依赖
    required_packages = ['streamlit', 'pandas', 'numpy', 'joblib', 'plotly']
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} 可用")
        except ImportError:
            issues.append(f"❌ 缺少依赖包: {package}")
    
    # 检查特征数据
    try:
        template = create_feature_template()
        print("✅ 特征模板创建正常")
    except Exception as e:
        issues.append(f"❌ 特征模板错误: {str(e)}")
    
    if not issues:
        print("🎉 部署就绪检查通过！")
        return True, []
    else:
        print("⚠️ 发现以下问题：")
        for issue in issues:
            print(f"  {issue}")
        return False, issues

if __name__ == "__main__":
    # 运行部署检查
    check_deployment_readiness()
