#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
铅中毒预测模型工具库：
- 提供模型加载和预测功能
- 实现结果解释和可视化
- 提供临床解读接口
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pickle
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

def load_saved_model(model_path='lead_poisoning_model.pkl'):
    """加载保存的模型和相关信息"""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        selected_features = model_data['selected_features']
        
        print(f"模型加载成功: {model_path}")
        print(f"模型类型: {type(model).__name__}")
        print(f"需要特征数量: {len(selected_features)}")
        
        return model, selected_features
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return None, None

def preprocess_new_data(data, selected_features, scaler=None):
    """预处理新数据以适配模型"""
    # 确保数据只包含需要的特征
    if isinstance(data, pd.DataFrame):
        # 检查是否包含所有必需特征
        missing_features = [f for f in selected_features if f not in data.columns]
        if missing_features:
            print(f"警告: 缺少所需特征: {missing_features}")
            return None
        
        # 按需提取特征
        X = data[selected_features].copy()
    else:
        # 如果是numpy数组或列表
        X = pd.DataFrame(data, columns=selected_features)
    
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

def predict_risk(model, data, selected_features, scaler=None, threshold=0.5):
    """预测铅中毒多次住院风险"""
    # 预处理数据
    X_scaled, _ = preprocess_new_data(data, selected_features, scaler)
    if X_scaled is None:
        return None, None
    
    # 预测
    try:
        # 获取概率预测
        y_proba = model.predict_proba(X_scaled)[:, 1]
        # 根据阈值进行分类
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

def explain_prediction(model, data, selected_features, scaler=None):
    """使用SHAP解释模型预测"""
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
                # 创建一个虚拟的SHAP值数据结构
                class DummyShapValues:
                    def __init__(self, values, feature_names):
                        self.values = values
                        self.feature_names = feature_names
                
                # 构建模拟的SHAP值
                values = np.zeros((len(X_scaled), len(selected_features), 1))
                for i in range(len(X_scaled)):
                    for j, imp in enumerate(importances):
                        values[i, j, 0] = imp * X_scaled[i, j]
                
                dummy_shap_values = DummyShapValues(values, selected_features)
                
                # 绘制基础特征重要性图
                plt.figure(figsize=(12, 8))
                feature_importance = pd.DataFrame({
                    '特征': selected_features,
                    '重要性': importances
                }).sort_values('重要性', ascending=False)
                
                sns.barplot(x='重要性', y='特征', data=feature_importance.head(15))
                plt.title('特征重要性')
                plt.tight_layout()
                plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                return dummy_shap_values
            else:
                print("模型没有feature_importances_属性，无法提供特征重要性解释")
                return None
        except Exception as e:
            print(f"特征重要性分析失败: {str(e)}")
            return None
    
    # 使用SHAP解释器
    try:
        # 创建解释器对象
        explainer = shap.Explainer(model, pd.DataFrame(X_scaled, columns=selected_features))
        # 计算SHAP值
        shap_values = explainer(pd.DataFrame(X_scaled, columns=selected_features))
        
        # 绘制SHAP摘要图
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, pd.DataFrame(X_scaled, columns=selected_features))
        plt.tight_layout()
        plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 单个预测SHAP力图
        if len(X_scaled) == 1:
            plt.figure(figsize=(12, 6))
            shap.plots.waterfall(shap_values[0], show=False)
            plt.title('预测影响因素分析')
            plt.tight_layout()
            plt.savefig('shap_waterfall.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        return shap_values
    except Exception as e:
        print(f"解释预测失败: {str(e)}")
        return None

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

def generate_patient_report(patient_data, risk_score, shap_values=None, selected_features=None):
    """生成病人个性化报告"""
    # 基本信息部分
    report = ["## 铅中毒风险评估报告\n"]
    
    # 添加患者基本信息
    report.append("### 患者信息")
    
    # 这里可以添加患者基本信息，如有
    if isinstance(patient_data, pd.DataFrame) and len(patient_data) == 1:
        for col in ['姓名', '年龄', '性别', 'ID', '住院号']:
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
            feature_importance = pd.DataFrame({
                '特征': selected_features,
                'SHAP值': np.abs(shap_values.values[0, :, 0])
            }).sort_values('SHAP值', ascending=False)
            
            # 添加前5个最重要的特征
            for i, (feature, importance) in enumerate(zip(feature_importance['特征'].head(5), 
                                                        feature_importance['SHAP值'].head(5)), 1):
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

def batch_predict(model, data, selected_features, scaler=None):
    """批量预测多个患者的风险"""
    # 确保数据只包含需要的特征
    if not isinstance(data, pd.DataFrame):
        print("错误: 批量预测需要DataFrame格式的数据")
        return None
    
    # 提取ID列（如果有）
    id_cols = [col for col in data.columns if any(id_key in col.lower() for id_key in ['id', '编号', '住院号'])]
    patient_ids = data[id_cols].copy() if id_cols else pd.DataFrame({'样本ID': range(len(data))})
    
    # 预处理数据
    X_scaled, _ = preprocess_new_data(data, selected_features, scaler)
    if X_scaled is None:
        return None
    
    # 预测
    try:
        # 获取概率预测
        y_proba = model.predict_proba(X_scaled)[:, 1]
        # 根据阈值进行分类
        y_pred = (y_proba >= 0.5).astype(int)
        
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