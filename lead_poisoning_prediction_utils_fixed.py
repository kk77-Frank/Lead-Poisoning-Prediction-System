#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修复版铅中毒预测模型工具库：
- 提供兼容的模型加载和预测功能
- 修复数据类型转换问题
- 实现结果解释和可视化
- 提供临床解读接口
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pickle
import joblib
import warnings
warnings.filterwarnings('ignore')

# 条件导入SHAP库
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("警告: SHAP 库未安装，将使用基础特征重要性方法")
    SHAP_AVAILABLE = False

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 真实的特征名称列表（从模型中提取）
REAL_FEATURE_NAMES = [
    "丙氨酸氨基转移酶（谷丙转氨酶）_to_间接胆红素_ratio",
    "腹痛",
    "入院时血铅水平（umol/L）",
    "丙氨酸氨基转移酶（谷丙转氨酶）_to_第1次血铅_ratio",
    "总胆红素",
    "本次住院期间使用的解毒剂总量（g）",
    "铅中毒风险评分",
    "血红蛋白（g/L）_to_第1次测血铅时间_天数差_ratio",
    "血红蛋白（g/L）_to_总胆红素_ratio",
    "平均红细胞体积（fL）_to_治疗后随访血铅监测次数_ratio",
    "血钙（mmol/L）_to_第1次测血铅时间_天数差_ratio",
    "血钙（mmol/L）",
    "入院时血铅水平（umol/L）_to_第1次测血铅时间_季节_ratio",
    "间接胆红素_to_第1次测血铅时间_天数差_ratio",
    "入院时血铅水平（umol/L）_to_第2次血铅_ratio",
    "入院时血铅水平（umol/L）_to_红细胞分布宽度变异系数%_ratio",
    "入院时血铅水平（umol/L）_to_第1次测血铅时间_天数差_ratio",
    "腹部压痛",
    "血红蛋白（g/L）",
    "红细胞压积%",
    "铅累积比_平均红细胞体积（fL）_to_第1次血铅_ratio",
    "poly_入院时血铅水平（umol/L）_第1次血铅",
    "入院时血铅水平（umol/L）_to_治疗后随访血铅监测次数_ratio",
    "住宅附近有无铅作业工厂",
    "铅累积比_天门冬氨酸氨基转移酶（谷草转氨酶）_to_第1次血铅_ratio",
    "天门冬氨酸氨基转移酶（谷草转氨酶）_to_第1次血铅_ratio",
    "超WHO倍数",
    "超WHO阈值",
    "入院时血铅水平（umol/L）_log",
    "poly_入院时血铅水平（umol/L）_治疗后随访血铅监测次数",
    "铅累积比_第2次血铅_log",
    "poly_血红蛋白（g/L）_血钙（mmol/L）",
    "血红蛋白（g/L）_to_血钙（mmol/L）_ratio",
    "总胆红素_to_第1次测血铅时间_天数差_ratio",
    "入院时血铅水平（umol/L）_to_第1次血铅_ratio",
    "poly_血钙（mmol/L）_治疗后随访血铅监测次数",
    "入院时血铅水平（umol/L）_to_血红蛋白（g/L）_ratio",
    "铅累积比_血钙（mmol/L）_to_第1次血铅_ratio",
    "头发稀少"
]

def get_feature_names():
    """获取真实特征名称列表"""
    return REAL_FEATURE_NAMES.copy()

def load_saved_model(model_path='lead_poisoning_optimized_model.pkl'):
    """兼容加载保存的模型和相关信息"""
    try:
        # 首先尝试用joblib加载
        try:
            model_data = joblib.load(model_path)
        except:
            # 如果joblib失败，尝试用pickle
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
        
        # 判断加载的数据类型
        if isinstance(model_data, dict):
            # 如果是字典格式，包含模型和其他信息
            model = model_data.get('model')
            selected_features = model_data.get('selected_features', [])
            scaler = model_data.get('scaler', None)
            optimal_threshold = model_data.get('optimal_threshold', 0.5)
            
            print(f"模型加载成功: {model_path}")
            print(f"模型类型: {type(model).__name__}")
            print(f"需要特征数量: {len(selected_features)}")
            print(f"最优阈值: {optimal_threshold}")
            
            return model, selected_features, scaler, optimal_threshold
        else:
            # 如果直接是模型对象
            model = model_data
            selected_features = []  # 无特征信息
            scaler = None
            optimal_threshold = 0.5
            
            print(f"模型加载成功: {model_path}")
            print(f"模型类型: {type(model).__name__}")
            print("警告: 无特征信息，将使用默认设置")
            
            return model, selected_features, scaler, optimal_threshold
            
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return None, None, None, None

def safe_numeric_conversion(data, exclude_columns=None):
    """安全的数值转换，排除身份信息列"""
    if exclude_columns is None:
        exclude_columns = ['Name', 'name', '姓名', 'ID', 'Patient_ID', '患者ID', '住院号', 'Hospital_ID']
    
    if isinstance(data, pd.DataFrame):
        numeric_data = data.copy()
        numeric_columns = []
        
        for col in data.columns:
            if col not in exclude_columns:
                try:
                    # 尝试转换为数值
                    pd.to_numeric(data[col])
                    numeric_columns.append(col)
                except:
                    # 如果转换失败，跳过这一列
                    print(f"警告: 列 '{col}' 不能转换为数值，将被跳过")
                    continue
        
        return numeric_data[numeric_columns]
    else:
        # 如果是数组或列表，直接转换
        return np.array(data, dtype=float)

def preprocess_new_data(data, selected_features=None, scaler=None, target_features=39):
    """预处理新数据以适配模型"""
    try:
        # 安全的数值转换
        if isinstance(data, pd.DataFrame):
            numeric_data = safe_numeric_conversion(data)
        else:
            numeric_data = pd.DataFrame(data)
        
        # 如果有特征名列表，尝试匹配
        if selected_features and len(selected_features) > 0:
            available_features = [f for f in selected_features if f in numeric_data.columns]
            if len(available_features) > 0:
                X = numeric_data[available_features].copy()
            else:
                print("警告: 没有找到匹配的特征列，使用所有数值列")
                X = numeric_data.copy()
        else:
            X = numeric_data.copy()
        
        # 确保有足够的特征
        if X.shape[1] < target_features:
            # 如果特征不够，用0填充
            missing_count = target_features - X.shape[1]
            for i in range(missing_count):
                X[f'feature_{X.shape[1] + i}'] = 0.0
            print(f"警告: 特征数量不足，已用0填充到{target_features}个特征")
        elif X.shape[1] > target_features:
            # 如果特征过多，只取前target_features个
            X = X.iloc[:, :target_features]
            print(f"警告: 特征数量过多，已截取前{target_features}个特征")
        
        # 处理缺失值
        for col in X.columns:
            if X[col].isnull().sum() > 0:
                X[col] = X[col].fillna(X[col].median())
        
        # 标准化数据
        if scaler is None:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = scaler.transform(X)
        
        return X_scaled, scaler
    
    except Exception as e:
        print(f"数据预处理失败: {str(e)}")
        return None, None

def predict_risk(model, data, selected_features=None, scaler=None, threshold=0.5):
    """预测铅中毒多次住院风险"""
    try:
        # 检查模型是否有predict_proba方法
        if not hasattr(model, 'predict_proba'):
            print("错误: 模型没有predict_proba方法")
            return None, None
        
        # 预处理数据
        X_scaled, _ = preprocess_new_data(data, selected_features, scaler)
        if X_scaled is None:
            return None, None
        
        # 预测
        y_proba = model.predict_proba(X_scaled)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        
        # 创建结果数据框
        results = pd.DataFrame({
            '风险概率': y_proba,
            '风险预测': y_pred
        })
        
        return results, y_proba
    
    except Exception as e:
        print(f"预测失败: {str(e)}")
        return None, None

def get_clinical_interpretation(risk_score, threshold=0.5):
    """根据风险评分给出临床解释"""
    if risk_score >= 0.8:
        risk_level = "极高风险"
        suggestion = "需立即采取干预措施，考虑积极治疗方案"
    elif risk_score >= threshold:
        risk_level = "高风险"
        suggestion = "建议密切监测，考虑预防性治疗"
    elif risk_score >= 0.3:
        risk_level = "中等风险"
        suggestion = "定期随访，加强健康教育"
    else:
        risk_level = "低风险"
        suggestion = "常规管理，注意铅暴露风险因素"
    
    interpretation = {
        "风险等级": risk_level,
        "风险概率": f"{risk_score:.2%}",
        "临床建议": suggestion
    }
    
    return interpretation

def explain_prediction(model, data, selected_features=None, scaler=None):
    """使用SHAP解释模型预测"""
    try:
        # 预处理数据
        X_scaled, _ = preprocess_new_data(data, selected_features, scaler)
        if X_scaled is None:
            return None
        
        # 检查是否可用SHAP
        if not SHAP_AVAILABLE:
            print("SHAP库未安装，将使用基础特征重要性方法")
            try:
                # 如果模型有feature_importances_属性，使用这个作为替代
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    # 创建特征重要性图
                    feature_names = selected_features if selected_features else [f'Feature_{i}' for i in range(len(importances))]
                    
                    plt.figure(figsize=(12, 8))
                    feature_importance = pd.DataFrame({
                        '特征': feature_names[:len(importances)],
                        '重要性': importances
                    }).sort_values('重要性', ascending=False)
                    
                    sns.barplot(x='重要性', y='特征', data=feature_importance.head(15))
                    plt.title('特征重要性')
                    plt.tight_layout()
                    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    return importances
                else:
                    print("模型没有feature_importances_属性，无法提供特征重要性解释")
                    return None
            except Exception as e:
                print(f"特征重要性分析失败: {str(e)}")
                return None
        
        # 使用SHAP解释器
        feature_names = selected_features if selected_features else [f'Feature_{i}' for i in range(X_scaled.shape[1])]
        explainer = shap.Explainer(model, pd.DataFrame(X_scaled, columns=feature_names))
        shap_values = explainer(pd.DataFrame(X_scaled, columns=feature_names))
        
        # 绘制SHAP摘要图
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, pd.DataFrame(X_scaled, columns=feature_names))
        plt.tight_layout()
        plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return shap_values
    
    except Exception as e:
        print(f"解释预测失败: {str(e)}")
        return None

def generate_patient_report(patient_data, risk_score, shap_values=None, selected_features=None):
    """生成病人个性化报告"""
    # 基本信息部分
    report = ["## 铅中毒风险评估报告\n"]
    
    # 添加患者基本信息
    report.append("### 患者信息")
    
    # 这里可以添加患者基本信息，如有
    if isinstance(patient_data, pd.DataFrame) and len(patient_data) == 1:
        for col in ['姓名', '年龄', '性别', 'Name', 'Age', 'Gender', 'ID', '住院号']:
            if col in patient_data.columns:
                report.append(f"- {col}: {patient_data[col].values[0]}")
    
    # 添加风险评估结果
    report.append("\n### 风险评估结果")
    interpretation = get_clinical_interpretation(risk_score)
    report.append(f"- 风险等级: {interpretation['风险等级']}")
    report.append(f"- 风险概率: {interpretation['风险概率']}")
    report.append(f"- 临床建议: {interpretation['临床建议']}")
    
    # 添加关键风险因素
    if shap_values is not None and selected_features is not None:
        report.append("\n### 关键风险因素")
        try:
            # 获取SHAP值的绝对值大小排序
            if hasattr(shap_values, 'values'):
                feature_importance = pd.DataFrame({
                    '特征': selected_features,
                    'SHAP值': np.abs(shap_values.values[0, :, 0])
                }).sort_values('SHAP值', ascending=False)
            else:
                # 如果是基础特征重要性
                feature_importance = pd.DataFrame({
                    '特征': selected_features[:len(shap_values)],
                    '重要性': shap_values
                }).sort_values('重要性', ascending=False)
            
            # 添加前5个最重要的特征
            for i, (feature, importance) in enumerate(zip(feature_importance.iloc[:5, 0], 
                                                        feature_importance.iloc[:5, 1]), 1):
                report.append(f"{i}. {feature}: {importance:.4f}")
        except:
            report.append("无法生成详细的特征重要性分析")
    
    # 添加建议和后续措施
    report.append("\n### 预防与干预建议")
    if interpretation['风险等级'] in ["高风险", "极高风险"]:
        report.append("1. 排查铅暴露源，移除潜在污染源")
        report.append("2. 考虑螯合剂治疗（DMSA、CaNa2EDTA等）")
        report.append("3. 加强营养支持，增加富含钙、铁、锌的食物摄入")
        report.append("4. 安排定期血铅水平监测（建议每1-3个月一次）")
    else:
        report.append("1. 定期监测血铅水平（建议每3-6个月一次）")
        report.append("2. 加强健康教育，避免接触铅污染源")
        report.append("3. 均衡饮食，保证充足的微量元素摄入")
    
    # 生成完整报告
    full_report = "\n".join(report)
    
    # 保存报告到文件
    with open("patient_risk_report.md", "w", encoding="utf-8") as f:
        f.write(full_report)
    
    return full_report

def batch_predict(model, data, selected_features=None, scaler=None, threshold=0.5):
    """批量预测多个患者的风险"""
    try:
        # 确保数据是DataFrame格式
        if not isinstance(data, pd.DataFrame):
            print("错误: 批量预测需要DataFrame格式的数据")
            return None
        
        # 检查模型是否有predict_proba方法
        if not hasattr(model, 'predict_proba'):
            print("错误: 模型没有predict_proba方法")
            return None
        
        # 提取ID列（如果有）
        id_columns = ['Name', 'name', '姓名', 'ID', 'Patient_ID', '患者ID', '住院号', 'Hospital_ID']
        available_id_cols = [col for col in data.columns if col in id_columns]
        patient_ids = data[available_id_cols].copy() if available_id_cols else pd.DataFrame({'样本ID': range(len(data))})
        
        # 预处理数据
        X_scaled, _ = preprocess_new_data(data, selected_features, scaler)
        if X_scaled is None:
            return None
        
        # 预测
        y_proba = model.predict_proba(X_scaled)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        
        # 添加解释
        risk_levels = []
        suggestions = []
        
        for score in y_proba:
            interp = get_clinical_interpretation(score)
            risk_levels.append(interp['风险等级'])
            suggestions.append(interp['临床建议'])
        
        # 创建结果数据框
        results = pd.concat([
            patient_ids,
            pd.DataFrame({
                '风险概率': y_proba,
                '风险预测': y_pred,
                '风险等级': risk_levels,
                '临床建议': suggestions
            })
        ], axis=1)
        
        # 保存预测结果
        results.to_csv('batch_prediction_results.csv', index=False, encoding='utf-8-sig')
        print(f"批量预测完成，共 {len(results)} 条记录")
        print(f"结果已保存至: batch_prediction_results.csv")
        
        return results
    
    except Exception as e:
        print(f"批量预测失败: {str(e)}")
        return None

# 为了兼容性，保持原有函数名
def load_saved_model_simple(model_path='lead_poisoning_optimized_model.pkl'):
    """简化版模型加载函数，仅返回模型和特征"""
    model, selected_features, scaler, threshold = load_saved_model(model_path)
    return model, selected_features
