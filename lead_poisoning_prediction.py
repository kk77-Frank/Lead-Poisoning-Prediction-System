#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
铅中毒预测模型：根据临床指标预测患者是否为多次住院（铅中毒复发）
- 分类标准: 第1次住院的作为样本0，第2次及以上的作为样本1
- 特征选择: 使用p值分析选择有统计显著性的特征
- 模型比较: 使用五种经典机器学习模型，选择最优模型
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing import StandardScaler, LabelEncoder, PowerTransformer, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, RFE, mutual_info_classif, SelectFromModel
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import CalibrationDisplay
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import clone

# 条件导入，确保即使没有某些库也能运行
# XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("警告: XGBoost 库未安装，将不会使用XGBoost模型")
    XGBOOST_AVAILABLE = False

# CatBoost
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    print("警告: CatBoost 库未安装，将不会使用CatBoost模型")
    CATBOOST_AVAILABLE = False

# LightGBM
try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("警告: LightGBM 库未安装，将不会使用LightGBM模型")
    LIGHTGBM_AVAILABLE = False

# 贝叶斯优化库
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Categorical, Integer
    BAYES_SEARCH_AVAILABLE = True
except ImportError:
    print("警告: scikit-optimize 库未安装，将使用随机搜索代替贝叶斯优化")
    BAYES_SEARCH_AVAILABLE = False

# 高级采样技术
try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
    from imblearn.combine import SMOTETomek, SMOTEENN
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBALANCED_AVAILABLE = True
except ImportError:
    print("警告: imbalanced-learn 库未安装，将使用类权重代替高级采样")
    IMBALANCED_AVAILABLE = False

# 高级解释性库
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("警告: SHAP 库未安装，将使用基础特征重要性方法")
    SHAP_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 设置随机种子
np.random.seed(42)

# 1. 数据加载与预处理
def load_data(file_path):
    """加载并预处理数据"""
    print("1. 数据加载与预处理")
    
    # 加载数据
    print(f"正在加载数据: {file_path}")
    df = pd.read_excel(file_path)
    
    # 显示基本信息
    print(f"\n数据集基本信息: {df.shape[0]}行 x {df.shape[1]}列")
    print("\n前5行数据:")
    print(df.head())
    
    # 查看缺失值情况
    missing_values = df.isnull().sum()
    print("\n各特征缺失值数量:")
    print(missing_values[missing_values > 0])
    
    # 检查数据中是否存在目标变量相关列
    potential_target_cols = [col for col in df.columns if '住院' in str(col) or '次' in str(col)]
    print("\n可能的目标变量列:")
    for col in potential_target_cols:
        print(f"- {col}")
    
    # 数据类型检查
    print("\n数据类型检查:")
    dtypes = df.dtypes.value_counts()
    for dtype, count in dtypes.items():
        print(f"- {dtype}: {count}列")
    
    return df

# 2. 特征工程和数据清洗
def preprocess_data(df):
    """特征工程和数据清洗"""
    print("\n2. 特征工程与数据清洗")
    
    # 复制数据，避免修改原始数据
    df_processed = df.copy()
    
    # 确认目标变量
    if "第几次住院" in df_processed.columns:
        # 创建目标变量: 1次住院为0，大于1次为1
        df_processed['target'] = df_processed['第几次住院'].apply(lambda x: 0 if x == 1 else 1)
        print("\n目标变量分布:")
        print(df_processed['target'].value_counts())
        print(f"类别0(首次住院)比例: {df_processed['target'].value_counts()[0]/len(df_processed):.2%}")
        print(f"类别1(多次住院)比例: {df_processed['target'].value_counts()[1]/len(df_processed):.2%}")
    elif 'target' not in df_processed.columns:
        print("警告: 未找到'第几次住院'列，无法创建目标变量")
        return None, None, None, None
    
    # 分离目标变量和特征
    y = df_processed['target']
    
    # 移除不需要的列(ID、姓名、住院次数等非预测特征)
    cols_to_drop = []
    for col in df_processed.columns:
        # 排除明显的非预测列
        if any(keyword in str(col).lower() for keyword in ['id', '编号', '姓名', 'name', '住院次数', 'target', '第几次住院']):
            cols_to_drop.append(col)
        # 排除缺失值过多的列(超过70%的缺失值)
        elif df_processed[col].isnull().sum() > 0.7 * len(df_processed):
            cols_to_drop.append(col)
            print(f"列 '{col}' 缺失值比例超过70%，将被移除")
    
    print(f"移除的非预测特征: {cols_to_drop}")
    X = df_processed.drop(columns=cols_to_drop, errors='ignore')
    
    # 处理日期时间类型列：提取时间特征
    datetime_cols = X.select_dtypes(include=['datetime64']).columns
    if len(datetime_cols) > 0:
        print("\n处理日期时间特征...")
        for col in datetime_cols:
            # 提取基本时间特征
            if '测血铅时间' in col:
                # 计算血铅检测时间与入院时间的差值(天数)
                try:
                    # 假设第一次测血铅时间接近入院时间
                    if '第1次测血铅时间' in X.columns:
                        base_date = X['第1次测血铅时间']
                        X[f'{col}_天数差'] = (X[col] - base_date).dt.days
                        
                        # 添加额外的时间特征：血铅检测间隔变化
                        if col not in ['第1次测血铅时间']:
                            # 计算测量间隔
                            col_num = int(col.replace('第','').replace('次测血铅时间',''))
                            if col_num > 1 and f'第{col_num-1}次测血铅时间' in X.columns:
                                prev_col = f'第{col_num-1}次测血铅时间'
                                X[f'{col}_与上次间隔'] = (X[col] - X[prev_col]).dt.days
                                print(f"  创建了时间间隔特征: {col}_与上次间隔")
                        
                        # 计算季节特征（可能与铅暴露相关）
                        X[f'{col}_季节'] = X[col].dt.quarter
                        print(f"  从 {col} 创建了季节特征")
                    
                    # 删除原始日期列
                    X = X.drop(columns=[col])
                    print(f"  从 {col} 创建了时间差特征")
                except Exception as e:
                    print(f"  处理时间特征 {col} 时出错: {str(e)}")
                    X = X.drop(columns=[col])
    
    # 处理分类变量
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    
    print(f"\n分类变量数量: {len(cat_cols)}")
    print(f"数值变量数量: {len(num_cols)}")
    
    # 使用高级缺失值插补 - IterativeImputer (多重插补)
    print("\n使用高级方法填充缺失值...")
    # 对数值特征使用迭代插补
    if len(num_cols) > 0:
        # 创建插补器
        imp_num = IterativeImputer(max_iter=10, random_state=42)
        # 保存列名以便后续恢复
        num_cols_list = list(num_cols)
        # 执行插补
        X_num_imputed = imp_num.fit_transform(X[num_cols])
        # 将插补结果放回DataFrame
        X_num = pd.DataFrame(X_num_imputed, columns=num_cols_list, index=X.index)
        # 替换原始列
        for col in num_cols:
            X[col] = X_num[col]
        print(f"  迭代插补处理了 {len(num_cols)} 个数值特征")
    
    # 对分类特征用众数填充
    for col in cat_cols:
        if X[col].isnull().sum() > 0:
            X[col] = X[col].fillna(X[col].mode()[0])
    
    # 检查并修复混合类型列
    mixed_type_cols = []
    for col in X.columns:
        if X[col].dtype == 'object':
            # 检查是否有混合类型
            types = set(type(x) for x in X[col].dropna().values)
            if len(types) > 1:
                mixed_type_cols.append(col)
                # 统一转换为字符串
                X[col] = X[col].astype(str)
    
    if mixed_type_cols:
        print(f"\n检测到{len(mixed_type_cols)}个混合类型列，已统一转换为字符串类型:")
        for col in mixed_type_cols[:5]:  # 只显示前5个
            print(f"- {col}")
        if len(mixed_type_cols) > 5:
            print(f"- ... 以及 {len(mixed_type_cols) - 5} 个其他列")
    
    # 【增强】添加医学专业知识的特征变换
    print("\n添加专业医学特征变换...")
    
    # 1. 体重指数（如果存在相关特征）
    if '体重' in X.columns and '身高' in X.columns:
        try:
            # BMI = 体重(kg) / (身高(m))^2
            X['BMI'] = X['体重'] / ((X['身高']/100) ** 2)
            print("  创建了BMI指标")
        except Exception as e:
            print(f"  创建BMI指标时出错: {str(e)}")
    
    # 2. 对关键医学指标应用对数变换（解决偏态分布）
    for col in num_cols:
        if any(term in col for term in ['血铅', '转氨酶']):
            try:
                # 确保数值为正
                if (X[col] > 0).all():
                    X[f'{col}_log'] = np.log(X[col])
                    print(f"  对 {col} 应用了对数变换")
            except Exception as e:
                print(f"  对 {col} 应用对数变换时出错: {str(e)}")
    
    # 3. 对血铅/年龄比例（调整年龄敏感性）
    if '入院时血铅水平（umol/L）' in X.columns and '年龄' in X.columns:
        try:
            X['血铅年龄比'] = X['入院时血铅水平（umol/L）'] / (X['年龄'] + 1)  # +1避免除零
            print("  创建了血铅年龄比特征")
        except Exception as e:
            print(f"  创建血铅年龄比特征时出错: {str(e)}")
    
    # 创建特征交互项（仅针对重要医学指标）
    key_medical_indicators = [col for col in num_cols if any(term in col for term in 
                             ['血铅', '红细胞', '血红蛋白', '血钙', '转氨酶', '胆红素'])]
    
    if len(key_medical_indicators) >= 2:
        print("\n创建重要医学指标的交互特征...")
        # 创建一些有医学意义的交互特征
        for i in range(len(key_medical_indicators)):
            for j in range(i+1, len(key_medical_indicators)):
                if i != j:
                    col1 = key_medical_indicators[i]
                    col2 = key_medical_indicators[j]
                    # 创建比率特征
                    interaction_name = f"{col1}_to_{col2}_ratio"
                    try:
                        X[interaction_name] = X[col1] / (X[col2] + 1e-10)  # 避免除零
                        print(f"  创建了交互特征: {interaction_name}")
                    except Exception as e:
                        print(f"  创建交互特征时出错: {str(e)}")
    
    # 【新增】创建多项式特征，针对关键临床指标
    print("\n创建多项式特征...")
    important_clinical_indicators = [col for col in num_cols if any(term in col for term in 
                                   ['血铅', '血红蛋白', '血钙'])]
    
    if len(important_clinical_indicators) >= 1:
        try:
            # 选择前5个最重要的指标避免特征爆炸
            selected_indicators = important_clinical_indicators[:5]
            
            # 创建多项式特征
            poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
            poly_features = poly.fit_transform(X[selected_indicators])
            
            # 生成特征名称
            feature_names = poly.get_feature_names_out(selected_indicators)
            
            # 只保留交互特征（排除原始特征）
            interaction_features = feature_names[len(selected_indicators):]
            interaction_values = poly_features[:, len(selected_indicators):]
            
            # 添加到数据框
            for i, feature_name in enumerate(interaction_features):
                # 简化特征名
                simple_name = feature_name.replace(' ', '_')
                X[f"poly_{simple_name}"] = interaction_values[:, i]
                
            print(f"  创建了 {len(interaction_features)} 个多项式交互特征")
        except Exception as e:
            print(f"  创建多项式特征时出错: {str(e)}")
    
    # 【新增】针对血铅水平创建分位数分段特征
    if '入院时血铅水平（umol/L）' in X.columns:
        try:
            # 创建四分位数特征
            quantiles = [0, 0.25, 0.5, 0.75, 1.0]
            labels = ['极低', '低', '中', '高']
            X['血铅四分位组'] = pd.qcut(X['入院时血铅水平（umol/L）'], q=quantiles, labels=labels)
            print("  创建了血铅四分位组分类特征")
            
            # 转换为数值编码
            X['血铅四分位组'] = X['血铅四分位组'].cat.codes
            
            # 创建阈值指示特征 (WHO和CDC标准)
            # 假设铅中毒阈值为5 μg/dL ≈ 0.24 μmol/L
            X['超WHO阈值'] = (X['入院时血铅水平（umol/L）'] > 0.24).astype(int)
            print("  创建了超WHO阈值指示特征")
        except Exception as e:
            print(f"  创建血铅分位特征时出错: {str(e)}")
    
    # 转换分类变量
    print("\n编码分类变量...")
    le = LabelEncoder()
    for col in cat_cols:
        try:
            # 确保所有值为字符串
            X[col] = X[col].astype(str)
            X[col] = le.fit_transform(X[col])
        except Exception as e:
            print(f"  编码列 '{col}' 时出错: {str(e)}")
            # 如果编码失败，尝试其他方法或移除该列
            X = X.drop(columns=[col])
            cat_cols = cat_cols.drop(col)
    
    # 应用PowerTransformer转换数值特征，使其更接近正态分布
    print("\n转换数值特征使其更接近正态分布...")
    pt = PowerTransformer(method='yeo-johnson', standardize=True)
    num_cols_current = X.select_dtypes(include=['int64', 'float64']).columns
    if len(num_cols_current) > 0:
        try:
            X[num_cols_current] = pd.DataFrame(
                pt.fit_transform(X[num_cols_current].fillna(0)), 
                columns=num_cols_current,
                index=X.index
            )
        except Exception as e:
            print(f"  转换数值特征时出错: {str(e)}")
            # 退回到标准化方法
            scaler = StandardScaler()
            X[num_cols_current] = scaler.fit_transform(X[num_cols_current].fillna(0))
    
    print("\n数据预处理完成！")
    print(f"处理后特征维度: {X.shape}")
    
    # 如果特征数过多，先进行初步的特征筛选
    if X.shape[1] > 50:
        print("\n特征数量较多，进行初步筛选以减少维度...")
        # 使用互信息来选择与目标变量相关的特征
        selector = SelectKBest(mutual_info_classif, k=min(50, X.shape[1]))
        X_new = selector.fit_transform(X, y)
        # 获取选择的特征名
        selected_cols = X.columns[selector.get_support()]
        X = X[selected_cols]
        print(f"初步特征筛选后维度: {X.shape}")
    
    # 返回处理后的数据、目标变量和列类型
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(exclude=['int64', 'float64']).columns
    
    return X, y, num_cols, cat_cols

# 新增: 高级特征工程函数
def advanced_feature_engineering(X, y=None, medical_knowledge=True):
    """针对小样本铅中毒数据的高级特征工程"""
    print("\n执行高级医学特征工程...")
    
    # 复制数据以避免修改原始数据
    X_advanced = X.copy()
    
    # 1. 铅暴露严重程度指数
    if '入院时血铅水平（umol/L）' in X_advanced.columns and '年龄' in X_advanced.columns:
        # 年龄加权的铅暴露指数（年龄越小，相同铅水平危害越大）
        X_advanced['铅暴露严重度'] = X_advanced['入院时血铅水平（umol/L）'] * (10 / (X_advanced['年龄'] + 1))
        print("  创建了铅暴露严重度指标 (血铅水平加权年龄)")
        
    # 2. 铅蓄积特征（处理铅的累积效应）
    lead_cols = [col for col in X_advanced.columns if '血铅' in col and ('第' in col or '既往' in col)]
    if lead_cols and '入院时血铅水平（umol/L）' in X_advanced.columns:
        # 如果有既往血铅数据，计算累积效应
        for col in lead_cols:
            try:
                X_advanced[f'铅累积比_{col}'] = X_advanced['入院时血铅水平（umol/L）'] / (X_advanced[col] + 0.01)
                print(f"  创建了铅累积比特征: 铅累积比_{col}")
            except Exception as e:
                print(f"  创建铅累积比特征时出错: {str(e)}")
    
    # 3. 贫血相关综合指标（铅中毒常引起贫血）
    blood_cols = [col for col in X_advanced.columns if '血红蛋白' in col or '红细胞' in col]
    if len(blood_cols) >= 2:
        try:
            # 创建贫血综合评分
            X_advanced['贫血综合指数'] = X_advanced[blood_cols].mean(axis=1)
            print("  创建了贫血综合指数 (血红蛋白和红细胞指标均值)")
        except Exception as e:
            print(f"  创建贫血综合指数时出错: {str(e)}")
        
    # 4. 铅清除效率（治疗效果指标）
    if '入院时血铅水平（umol/L）' in X_advanced.columns and '出院时血铅水平' in X_advanced.columns:
        try:
            X_advanced['铅清除率'] = (X_advanced['入院时血铅水平（umol/L）'] - X_advanced['出院时血铅水平']) / (X_advanced['入院时血铅水平（umol/L）'] + 0.01)
            print("  创建了铅清除率指标 (入院与出院血铅差异比例)")
        except Exception as e:
            print(f"  创建铅清除率指标时出错: {str(e)}")
    
    # 5. WHO标准比值（比较危险程度）
    # WHO儿童铅中毒参考值约为0.24 μmol/L
    if '入院时血铅水平（umol/L）' in X_advanced.columns:
        try:
            X_advanced['超WHO倍数'] = X_advanced['入院时血铅水平（umol/L）'] / 0.24
            print("  创建了超WHO倍数指标 (相对WHO标准)")
        except Exception as e:
            print(f"  创建超WHO倍数指标时出错: {str(e)}")
        
    # 6. 铅中毒症状综合评分
    symptom_cols = [col for col in X_advanced.columns if col in ['腹痛', '头痛', '贫血', '呕吐', '发育迟缓']]
    if len(symptom_cols) >= 2:
        try:
            X_advanced['症状综合评分'] = X_advanced[symptom_cols].sum(axis=1)
            print(f"  创建了症状综合评分 (基于{len(symptom_cols)}个症状特征)")
        except Exception as e:
            print(f"  创建症状综合评分时出错: {str(e)}")
        
    # 7. 肝功能异常指数（铅中毒影响肝功能）
    liver_cols = [col for col in X_advanced.columns if '转氨酶' in col or '胆红素' in col]
    if len(liver_cols) >= 1:
        try:
            liver_values = X_advanced[liver_cols].fillna(0)
            X_advanced['肝功能异常指数'] = liver_values.mean(axis=1)
            print(f"  创建了肝功能异常指数 (基于{len(liver_cols)}个肝功能指标)")
        except Exception as e:
            print(f"  创建肝功能异常指数时出错: {str(e)}")
    
    # 8. 铅中毒风险评分（综合多因素）
    if '入院时血铅水平（umol/L）' in X_advanced.columns:
        risk_factors = []
        weights = []
        
        # 添加基本血铅因素
        risk_factors.append(X_advanced['入院时血铅水平（umol/L）'] / 0.24)  # 标准化为WHO标准倍数
        weights.append(1.0)
        
        # 添加年龄因素（若存在）
        if '年龄' in X_advanced.columns:
            # 年龄越小风险越高，使用反比
            risk_factors.append(10 / (X_advanced['年龄'] + 1))
            weights.append(0.7)
        
        # 添加症状数量因素（若存在）
        if len(symptom_cols) >= 2 and '症状综合评分' in X_advanced.columns:
            risk_factors.append(X_advanced['症状综合评分'] / len(symptom_cols))  # 归一化症状评分
            weights.append(0.5)
        
        # 计算加权风险评分
        if risk_factors:
            try:
                # 将各因素转换为numpy数组并标准化
                risk_array = np.column_stack([np.array(factor) for factor in risk_factors])
                # 应用权重
                weighted_risk = np.average(risk_array, axis=1, weights=weights)
                X_advanced['铅中毒风险评分'] = weighted_risk
                print(f"  创建了铅中毒风险评分 (基于{len(risk_factors)}个风险因素的加权评分)")
            except Exception as e:
                print(f"  创建铅中毒风险评分时出错: {str(e)}")
    
    # 9. 治疗响应性指标
    treatment_cols = [col for col in X_advanced.columns if '解毒剂' in col or '治疗' in col]
    if treatment_cols and '入院时血铅水平（umol/L）' in X_advanced.columns and '出院时血铅水平' in X_advanced.columns:
        try:
            # 血铅下降与治疗强度的比值
            lead_reduction = X_advanced['入院时血铅水平（umol/L）'] - X_advanced['出院时血铅水平']
            treatment_intensity = X_advanced[treatment_cols].mean(axis=1)
            X_advanced['治疗响应性'] = lead_reduction / (treatment_intensity + 0.01)
            print("  创建了治疗响应性指标 (血铅下降与治疗强度比值)")
        except Exception as e:
            print(f"  创建治疗响应性指标时出错: {str(e)}")
            
    # 10. 铅暴露持续时间评估
    exposure_cols = [col for col in X_advanced.columns if '暴露' in col or '接触' in col]
    if exposure_cols:
        try:
            X_advanced['铅暴露持续评分'] = X_advanced[exposure_cols].mean(axis=1)
            print("  创建了铅暴露持续评分")
        except Exception as e:
            print(f"  创建铅暴露持续评分时出错: {str(e)}")
            
    print(f"高级特征工程完成，创建了{X_advanced.shape[1] - X.shape[1]}个新特征")
    return X_advanced

# 3. 特征选择
def select_features(X, y, num_cols):
    """使用多种方法选择重要特征"""
    print("\n3. 高级特征选择")
    
    # 检查输入
    if X.shape[1] == 0:
        print("错误: 没有可用特征进行选择")
        return None, None
    
    try:
        # 特征选择结果存储
        feature_importance_dict = {}
        
        # 1. 方差膨胀因子(VIF)检测多重共线性
        print("\n多重共线性检测 (VIF):")
        vif_data = pd.DataFrame()
        vif_data["特征"] = X.columns
        
        # 计算VIF，处理可能的错误
        vif_values = []
        for i in range(X.shape[1]):
            try:
                vif = variance_inflation_factor(X.values, i)
                # 处理无穷大值
                if np.isinf(vif) or np.isnan(vif):
                    vif = 1000  # 设置一个较大的默认值
                vif_values.append(vif)
            except Exception:
                vif_values.append(1000)  # 出错时设置默认值
                
        vif_data["VIF"] = vif_values
        high_vif_features = vif_data[vif_data["VIF"] > 10]["特征"].tolist()
        print(f"检测到{len(high_vif_features)}个高度共线性特征(VIF>10)")
        print(vif_data.sort_values("VIF", ascending=False).head(10))
        
        # 2. 使用p值进行特征选择
        print("\n单变量特征选择 (基于p值):")
        
        # 使用更健壮的方法计算p值
        p_values = {}
        for col in X.columns:
            try:
                # 添加常数项
                X_col = sm.add_constant(X[col])
                # 拟合逻辑回归模型
                logit_model = sm.Logit(y, X_col)
                result = logit_model.fit(disp=0)
                # 获取特征的p值，跳过常数项
                p_values[col] = result.pvalues[1] if len(result.pvalues) > 1 else result.pvalues[0]
                feature_importance_dict[col] = {'p_value': p_values[col]}
            except Exception as e:
                print(f"计算 '{col}' 的p值时出错: {str(e)}")
                p_values[col] = 1.0  # 出错时设为1
                feature_importance_dict[col] = {'p_value': 1.0}
        
        # 转换为DataFrame便于查看
        p_df = pd.DataFrame({
            "特征": list(p_values.keys()), 
            "p值": list(p_values.values())
        })
        p_df = p_df.sort_values("p值")
        
        # 筛选p值小于0.05的特征
        sig_features = p_df[p_df["p值"] < 0.05]["特征"].tolist()
        
        print(f"\n统计显著特征 (p<0.05): {len(sig_features)}个")
        print(p_df.head(10))
        
        # 3. 使用随机森林特征重要性
        print("\n随机森林特征重要性评估:")
        try:
            # 创建和训练随机森林
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            # 提取特征重要性
            rf_importances = pd.DataFrame({
                '特征': X.columns,
                '重要性': rf.feature_importances_
            }).sort_values('重要性', ascending=False)
            
            print("随机森林特征重要性 (前10个):")
            print(rf_importances.head(10))
            
            # 将随机森林特征重要性添加到字典
            for feature, importance in zip(rf_importances['特征'], rf_importances['重要性']):
                if feature in feature_importance_dict:
                    feature_importance_dict[feature]['rf_importance'] = importance
                else:
                    feature_importance_dict[feature] = {'rf_importance': importance}
            
        except Exception as e:
            print(f"随机森林特征重要性评估出错: {str(e)}")
        
        # 4. 使用Lasso特征选择
        print("\nLasso特征选择:")
        try:
            lasso = Lasso(alpha=0.01, random_state=42)
            lasso.fit(X, y)
            
            # 提取Lasso系数
            lasso_coef = pd.DataFrame({
                '特征': X.columns,
                '系数': np.abs(lasso.coef_)
            }).sort_values('系数', ascending=False)
            
            print("Lasso特征重要性 (前10个):")
            print(lasso_coef.head(10))
            
            # 将Lasso系数添加到字典
            for feature, coef in zip(lasso_coef['特征'], lasso_coef['系数']):
                if feature in feature_importance_dict:
                    feature_importance_dict[feature]['lasso_coef'] = coef
                else:
                    feature_importance_dict[feature] = {'lasso_coef': coef}
            
            # 基于Lasso系数选择特征
            lasso_selected = lasso_coef[lasso_coef['系数'] > 0]['特征'].tolist()
            print(f"Lasso选择的特征: {len(lasso_selected)}个")
            
        except Exception as e:
            print(f"Lasso特征选择出错: {str(e)}")
            lasso_selected = []
        
        # 5. 互信息特征选择
        print("\n互信息特征选择:")
        try:
            # 计算互信息
            mi_scores = mutual_info_classif(X, y, random_state=42)
            mi_df = pd.DataFrame({
                '特征': X.columns,
                '互信息': mi_scores
            }).sort_values('互信息', ascending=False)
            
            print("互信息特征排名 (前10个):")
            print(mi_df.head(10))
            
            # 将互信息添加到字典
            for feature, mi in zip(mi_df['特征'], mi_df['互信息']):
                if feature in feature_importance_dict:
                    feature_importance_dict[feature]['mutual_info'] = mi
                else:
                    feature_importance_dict[feature] = {'mutual_info': mi}
            
            # 基于互信息选择特征 (选择前50%的特征)
            mi_threshold = np.median(mi_scores)
            mi_selected = mi_df[mi_df['互信息'] > mi_threshold]['特征'].tolist()
            print(f"基于互信息选择的特征: {len(mi_selected)}个")
            
        except Exception as e:
            print(f"互信息特征选择出错: {str(e)}")
            mi_selected = []
        
        # 6. 特征选择整合：创建集成排名
        print("\n特征集成排名:")
        feature_scores = {}
        
        for feature in feature_importance_dict:
            # 初始化分数
            score = 0
            # 添加p值评分 (p值越小越好)
            if 'p_value' in feature_importance_dict[feature]:
                p = feature_importance_dict[feature]['p_value']
                score += (1 - min(p, 1)) * 3  # p值权重为3
            
            # 添加随机森林重要性评分
            if 'rf_importance' in feature_importance_dict[feature]:
                rf_imp = feature_importance_dict[feature]['rf_importance']
                score += rf_imp * 2  # 随机森林权重为2
            
            # 添加Lasso系数评分
            if 'lasso_coef' in feature_importance_dict[feature]:
                lasso_c = feature_importance_dict[feature]['lasso_coef']
                # 将Lasso系数归一化为0-1之间
                max_coef = max([feature_importance_dict[f].get('lasso_coef', 0) for f in feature_importance_dict])
                if max_coef > 0:
                    score += (lasso_c / max_coef) * 2  # Lasso权重为2
            
            # 添加互信息评分
            if 'mutual_info' in feature_importance_dict[feature]:
                mi = feature_importance_dict[feature]['mutual_info']
                # 归一化互信息
                max_mi = max([feature_importance_dict[f].get('mutual_info', 0) for f in feature_importance_dict])
                if max_mi > 0:
                    score += (mi / max_mi) * 1  # 互信息权重为1
            
            # 存储最终分数
            feature_scores[feature] = score
        
        # 创建特征排名DataFrame
        ranking_df = pd.DataFrame({
            '特征': list(feature_scores.keys()),
            '集成分数': list(feature_scores.values())
        }).sort_values('集成分数', ascending=False)
        
        print("特征集成排名 (前15个):")
        print(ranking_df.head(15))
        
        # 选择特征
        # 1. 基于统计显著性
        # 2. 基于集成评分 (选择前20个或得分大于某个阈值的特征)
        
        # 合并多种特征选择方法的结果
        selected_features = []
        
        # 如果有显著特征，优先选择
        if len(sig_features) >= 5:
            selected_features.extend(sig_features)
            print(f"\n从统计显著性选择了 {len(sig_features)} 个特征")
        
        # 从集成排名中添加高分特征
        top_ranked_features = ranking_df.head(min(20, len(ranking_df)))['特征'].tolist()
        # 只添加不在sig_features中的特征
        additional_features = [f for f in top_ranked_features if f not in selected_features]
        selected_features.extend(additional_features)
        print(f"从集成排名添加了 {len(additional_features)} 个高评分特征")
        
        # 如果特征数仍然太少，考虑从Lasso和互信息中添加
        if len(selected_features) < 10:
            remaining_needed = 10 - len(selected_features)
            # 从Lasso选择中添加
            if lasso_selected:
                lasso_add = [f for f in lasso_selected if f not in selected_features][:remaining_needed]
                selected_features.extend(lasso_add)
                remaining_needed -= len(lasso_add)
                print(f"从Lasso选择添加了 {len(lasso_add)} 个特征")
            
            # 如果仍需要更多特征，从互信息中添加
            if remaining_needed > 0 and mi_selected:
                mi_add = [f for f in mi_selected if f not in selected_features][:remaining_needed]
                selected_features.extend(mi_add)
                print(f"从互信息选择添加了 {len(mi_add)} 个特征")
        
        # 最终确保至少有一些特征被选择
        if len(selected_features) < 5:
            print("警告: 特征选择结果不足5个特征，添加排名前10的特征")
            top10 = ranking_df.head(10)['特征'].tolist()
            selected_features = list(set(selected_features + top10))
        
        # 去除选择特征中的重复项
        selected_features = list(set(selected_features))
        
        print(f"\n最终选择的特征: {len(selected_features)}个")
        for i, feature in enumerate(selected_features[:15], 1):
            # 添加各种评分信息
            info = []
            if feature in feature_importance_dict:
                if 'p_value' in feature_importance_dict[feature]:
                    info.append(f"p值={feature_importance_dict[feature]['p_value']:.4f}")
                if 'rf_importance' in feature_importance_dict[feature]:
                    info.append(f"RF={feature_importance_dict[feature]['rf_importance']:.4f}")
                if 'lasso_coef' in feature_importance_dict[feature]:
                    info.append(f"Lasso={feature_importance_dict[feature]['lasso_coef']:.4f}")
            
            print(f"{i}. {feature} ({', '.join(info)})")
        
        if len(selected_features) > 15:
            print(f"... 以及 {len(selected_features)-15} 个其他特征")
        
        # 返回筛选后的特征矩阵
        X_selected = X[selected_features]
        
        return X_selected, selected_features
        
    except Exception as e:
        print(f"特征选择过程中出错: {str(e)}")
        # 如果所有选择方法都失败，则返回原始特征集的前10个
        print("使用前10个特征作为备选")
        selected_features = X.columns[:10].tolist()
        return X[selected_features], selected_features

# 4. 构建和评估模型
def build_models(X, y):
    """构建并评估多个机器学习模型，使用类权重处理类别不平衡"""
    print("\n4. 高级模型构建与评估")
    
    # 禁用SMOTE，改用class_weight处理类别不平衡
    global apply_smote
    apply_smote = False
    
    try:
        # 检查数据集大小，调整参数
        small_dataset = X.shape[0] < 100
        print(f"数据集样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
        
        # 分割训练集和测试集 - 使用分层抽样保持类别比例
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        print(f"\n训练集: {X_train.shape[0]}样本")
        print(f"测试集: {X_test.shape[0]}样本")
        
        # 检查类别不平衡情况
        class_counts = np.bincount(y)
        print(f"\n目标变量分布: 0类 - {class_counts[0]}样本, 1类 - {class_counts[1]}样本")
        imbalance_ratio = class_counts[0] / class_counts[1]
        print(f"类别不平衡比例: {imbalance_ratio:.2f}:1 (较多:较少)")
        
        # 计算类别权重
        weight_ratio = {
            0: 1.0, 
            1: imbalance_ratio  # 少数类权重更高
        }
        print(f"使用类别权重处理不平衡: {weight_ratio}")
        
        # 创建基础模型，使用类权重处理不平衡
        base_models = {
            "逻辑回归": LogisticRegression(random_state=42, max_iter=2000, class_weight=weight_ratio, solver='liblinear'),
            "随机森林": RandomForestClassifier(random_state=42, class_weight=weight_ratio),
            "支持向量机": SVC(probability=True, random_state=42, class_weight=weight_ratio),
            "K近邻": KNeighborsClassifier(), # KNN不支持类别权重
            "XGBoost": XGBClassifier(
                use_label_encoder=False, 
                eval_metric='logloss', 
                random_state=42,
                scale_pos_weight=imbalance_ratio  # XGBoost的类别权重参数
            ),
            "CatBoost": CatBoostClassifier(
                random_seed=42,
                verbose=0,
                class_weights=[1.0, imbalance_ratio]
            ),
            "LightGBM": LGBMClassifier(
                random_state=42,
                class_weight=weight_ratio,
                min_child_samples=5,    # 增加最小子节点样本数
                min_child_weight=1,     # 设置最小子节点权重
                min_split_gain=0.01,    # 设置最小分裂增益
                subsample=0.8,          # 子采样比例
                colsample_bytree=0.8,   # 特征采样比例
                reg_alpha=0.1,          # L1正则化
                reg_lambda=1.0,         # L2正则化
                verbose=-1              # 禁用冗余警告
            )
        }
        
        # 【新增】创建Stacking集成模型
        # 准备第一层模型
        estimators = [
            ('rf', RandomForestClassifier(random_state=42, class_weight=weight_ratio)),
            ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, scale_pos_weight=imbalance_ratio)),
            ('svc', SVC(probability=True, random_state=42, class_weight=weight_ratio))
        ]
        
        # 第二层模型（元模型）
        stacking_model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(random_state=42),
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            stack_method='predict_proba',
            n_jobs=1
        )
        
        # 添加Stacking模型到评估列表
        base_models["Stacking集成"] = stacking_model
        
        # 【优化】定义贝叶斯优化的参数空间
        bayes_params_spaces = {
            "逻辑回归": {
                'C': Real(1e-4, 1e2, prior='log-uniform')
            },
            "随机森林": {
                'n_estimators': Integer(50, 300),
                'max_depth': Integer(3, 20),
                'min_samples_split': Integer(2, 10),
                'min_samples_leaf': Integer(1, 10)
            },
            "支持向量机": {
                'C': Real(1e-3, 1e2, prior='log-uniform'),
                'gamma': Real(1e-4, 1e1, prior='log-uniform'),
                'kernel': Categorical(['linear', 'rbf', 'poly']),
                'degree': Integer(2, 3)
            },
            "K近邻": {
                'n_neighbors': Integer(1, 20),
                'weights': Categorical(['uniform', 'distance']),
                'p': Integer(1, 2)
            },
            "XGBoost": {
                'n_estimators': Integer(50, 500),
                'max_depth': Integer(3, 10),
                'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'subsample': Real(0.6, 1.0),
                'colsample_bytree': Real(0.6, 1.0)
            },
            "CatBoost": {
                'iterations': Integer(50, 300),
                'depth': Integer(4, 10),
                'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'l2_leaf_reg': Real(1, 10)
            },
            "LightGBM": {
                'n_estimators': Integer(50, 150),
                'max_depth': Integer(3, 8),
                'learning_rate': Real(0.01, 0.1),
                'min_child_samples': Integer(3, 10),
                'min_split_gain': Real(0.001, 0.05),
                'subsample': Real(0.7, 0.9),
                'colsample_bytree': Real(0.7, 0.9)
            }
        }
        
        # 精简参数空间（用于小数据集）
        if small_dataset:
            print("\n检测到小数据集，使用精简参数空间进行贝叶斯优化")
            bayes_params_spaces = {
                "逻辑回归": {
                    'C': Real(0.01, 10)
                },
                "随机森林": {
                    'n_estimators': Integer(50, 150),
                    'max_depth': Integer(3, 10)
                },
                "支持向量机": {
                    'C': Real(0.1, 10),
                    'kernel': Categorical(['linear', 'rbf'])
                },
                "K近邻": {
                    'n_neighbors': Integer(1, 10),
                    'weights': Categorical(['uniform', 'distance'])
                },
                "XGBoost": {
                    'n_estimators': Integer(50, 200),
                    'max_depth': Integer(3, 6),
                    'learning_rate': Real(0.01, 0.2)
                },
                "CatBoost": {
                    'iterations': Integer(50, 150),
                    'depth': Integer(4, 8)
                },
                "LightGBM": {
                    'n_estimators': Integer(50, 150),
                    'max_depth': Integer(3, 8)
                }
            }
            # Stacking模型不进行超参数优化
            bayes_params_spaces["Stacking集成"] = None
        else:
            print("\n使用完整参数空间进行贝叶斯优化")
            # Stacking模型不进行超参数优化
            bayes_params_spaces["Stacking集成"] = None
        
        # 存储模型评估结果
        results = {}
        best_models = {}
        
        # 使用分层K折交叉验证
        cv = StratifiedKFold(n_splits=min(5, X_train.shape[0] // 10 + 1), shuffle=True, random_state=42)
        
        # 训练和评估每个模型
        for name, model in base_models.items():
            try:
                print(f"\n训练模型: {name}")
                
                if name == "Stacking集成":
                    # Stacking模型直接训练，不进行超参数优化
                    print("Stacking集成模型使用默认配置训练...")
                    model.fit(X_train, y_train)
                    best_model = model
                else:
                    # 使用贝叶斯优化调整超参数
                    param_space = bayes_params_spaces[name]
                    
                    # 贝叶斯优化
                    print(f"对{name}执行贝叶斯参数优化...")
                    
                    # 确定搜索次数
                    n_iter = 10 if small_dataset else 30
                    
                    # 执行贝叶斯搜索
                    search = BayesSearchCV(
                        estimator=model,
                        search_spaces=param_space,
                        n_iter=n_iter,
                        scoring='roc_auc',
                        cv=cv,
                        n_jobs=1,
                        verbose=0,
                        random_state=42,
                        return_train_score=True
                    )
                    
                    # 执行参数搜索
                    search.fit(X_train, y_train)
                    
                    # 获取最佳模型
                    best_model = search.best_estimator_
                    
                    # 输出最佳参数和交叉验证结果
                    print(f"最佳参数: {search.best_params_}")
                    print(f"交叉验证AUC: {search.best_score_:.4f}")
                
                best_models[name] = best_model
                
                # 在测试集上评估
                y_pred = best_model.predict(X_test)
                try:
                    y_prob = best_model.predict_proba(X_test)[:, 1]
                except:
                    # 如果模型没有predict_proba方法，则使用decision_function
                    y_prob = best_model.decision_function(X_test)
                    # 转换为0-1范围的概率
                    y_prob = 1 / (1 + np.exp(-y_prob))
                
                # 计算性能指标
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                auc_score = roc_auc_score(y_test, y_prob)
                average_precision = average_precision_score(y_test, y_prob)
                
                # 保存结果
                results[name] = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "auc": auc_score,
                    "avg_precision": average_precision,
                    "y_pred": y_pred,
                    "y_prob": y_prob
                }
                
                print(f"准确率: {accuracy:.4f}")
                print(f"精确率: {precision:.4f}")
                print(f"召回率: {recall:.4f}")
                print(f"F1得分: {f1:.4f}")
                print(f"AUC: {auc_score:.4f}")
                print(f"AP(平均精确率): {average_precision:.4f}")
                
                # 显示混淆矩阵
                cm = confusion_matrix(y_test, y_pred)
                print("\n混淆矩阵:")
                print(cm)
                
                # 显示分类报告
                print("\n分类报告:")
                print(classification_report(y_test, y_pred, zero_division=0))
                
            except Exception as e:
                print(f"训练 {name} 模型时出错: {str(e)}")
                # 使用默认参数重试
                try:
                    print("使用默认参数重试...")
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    try:
                        y_prob = model.predict_proba(X_test)[:, 1]
                    except:
                        # 如果模型没有predict_proba方法
                        y_prob = np.where(y_pred == 1, 0.9, 0.1)  # 简单替代
                    
                    # 计算性能指标
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    auc_score = roc_auc_score(y_test, y_prob)
                    
                    # 保存结果
                    best_models[name] = model
                    results[name] = {
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "auc": auc_score,
                        "y_pred": y_pred,
                        "y_prob": y_prob
                    }
                    
                    print(f"默认参数模型评估结果:")
                    print(f"准确率: {accuracy:.4f}")
                    print(f"AUC: {auc_score:.4f}")
                    
                except Exception as e2:
                    print(f"使用默认参数训练失败: {str(e2)}")
        
        # 检查是否至少有一个模型成功训练
        if len(results) == 0:
            print("所有模型训练失败!")
            return None, None, None, None
            
        return results, best_models, X_test, y_test
    
    except Exception as e:
        print(f"模型构建过程中出错: {str(e)}")
        return None, None, None, None

# 辅助函数 - 将参数网格转换为分布
def get_param_distribution(model_name, param_grid):
    """将参数网格转换为参数分布字典，适用于RandomizedSearchCV"""
    param_dist = {}
    
    # 获取apply_smote变量，这个是在build_models中定义的全局变量
    # 检查是否已经定义，如果没有默认为False
    global apply_smote
    if 'apply_smote' not in globals():
        apply_smote = False
    
    if apply_smote:
        # 如果使用SMOTE，需要给参数加前缀
        prefix = 'model__'
    else:
        # 不使用SMOTE，无需前缀
        prefix = ''
    
    # 遍历参数网格
    for param, values in param_grid.items():
        param_name = f"{prefix}{param}"
        
        if isinstance(values, list):
            if all(isinstance(v, (int, float)) for v in values) and len(values) > 2:
                # 数值类型参数 - 保持列表形式
                param_dist[param_name] = values
            else:
                # 分类参数 - 保持列表形式
                param_dist[param_name] = values
        else:
            # 单值参数
            param_dist[param_name] = [values]
    
    return param_dist

# 5. 模型比较与可视化
def compare_models(results, best_models, X_test, y_test, selected_features):
    """比较不同模型的性能并可视化结果，提供更全面的评估"""
    print("\n5. 高级模型比较与可视化")
    
    # 检查输入是否有效
    if results is None or len(results) == 0 or best_models is None:
        print("无有效模型可供比较")
        return None, None
    
    try:
        # 创建性能指标比较表
        metrics_df = pd.DataFrame({
            "模型": list(results.keys()),
            "准确率": [results[model]["accuracy"] for model in results],
            "精确率": [results[model]["precision"] for model in results],
            "召回率": [results[model]["recall"] for model in results],
            "F1得分": [results[model]["f1"] for model in results],
            "AUC": [results[model]["auc"] for model in results],
            "平均精确率": [results[model].get("avg_precision", 0) for model in results]
        })
        
        # 按AUC排序
        metrics_df = metrics_df.sort_values("AUC", ascending=False)
        print("\n模型性能比较:")
        print(metrics_df)
        
        # 计算综合评分 - 平衡各个指标的影响
        metrics_df["综合评分"] = (
            metrics_df["AUC"] * 0.3 + 
            metrics_df["F1得分"] * 0.3 + 
            metrics_df["准确率"] * 0.2 + 
            metrics_df["精确率"] * 0.1 + 
            metrics_df["召回率"] * 0.1
        )
        
        # 按综合评分排序
        metrics_df = metrics_df.sort_values("综合评分", ascending=False)
        print("\n模型综合评分比较 (综合权重: AUC 30%, F1 30%, 准确率 20%, 精确率 10%, 召回率 10%):")
        print(metrics_df)
        
        # 找出最佳模型
        best_model_name = metrics_df.iloc[0]["模型"]
        best_model = best_models[best_model_name]
        
        print(f"\n最佳模型: {best_model_name}")
        print(f"综合评分: {metrics_df.iloc[0]['综合评分']:.4f}")
        print(f"AUC: {metrics_df.iloc[0]['AUC']:.4f}")
        print(f"F1得分: {metrics_df.iloc[0]['F1得分']:.4f}")
        
        # 绘制ROC曲线比较
        try:
            plt.figure(figsize=(14, 12))
            
            # 添加对角线参考线
            plt.plot([0, 1], [0, 1], 'k--', label='随机猜测 (AUC = 0.5)')
            
            # 为每个模型绘制ROC曲线
            # 使用色彩映射区分不同模型
            cmap = plt.cm.get_cmap('tab10', len(results))
            
            for i, (name, result) in enumerate(results.items()):
                fpr, tpr, _ = roc_curve(y_test, result["y_prob"])
                auc_score = result["auc"]
                
                # 使用不同的线型和颜色
                linestyle = '-' if i < 5 else '--'
                if name == best_model_name:
                    # 最佳模型使用较粗线条
                    plt.plot(fpr, tpr, linewidth=3, label=f'{name} (AUC = {auc_score:.4f})',
                            color=cmap(i), linestyle=linestyle)
                else:
                    plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc_score:.4f})',
                            color=cmap(i), linestyle=linestyle)
            
            # 美化图表
            plt.xlabel('假阳性率 (FPR)', fontsize=14)
            plt.ylabel('真阳性率 (TPR)', fontsize=14)
            plt.title('ROC曲线比较', fontsize=16, fontweight='bold')
            plt.legend(loc='lower right', fontsize=12, framealpha=0.8)
            plt.grid(True, alpha=0.3)
            
            # 限制坐标轴范围，聚焦在ROC曲线的有效部分
            plt.xlim([-0.01, 1.01])
            plt.ylim([-0.01, 1.01])
            
            # 添加次坐标轴标签
            plt.text(0.5, 0.03, '更低特异性', ha='center', va='center', fontsize=12)
            plt.text(0.5, 0.97, '更高特异性', ha='center', va='center', fontsize=12)
            plt.text(0.03, 0.5, '更低敏感性', ha='center', va='center', rotation=90, fontsize=12)
            plt.text(0.97, 0.5, '更高敏感性', ha='center', va='center', rotation=90, fontsize=12)
            
            plt.tight_layout()
            plt.savefig('roc_curves_comparison.png', dpi=300, bbox_inches='tight')
            print("\nROC曲线已保存为: roc_curves_comparison.png")
        except Exception as e:
            print(f"绘制ROC曲线时出错: {str(e)}")
        
        # 绘制精确率-召回率曲线
        try:
            plt.figure(figsize=(14, 12))
            
            # 为每个模型绘制PR曲线
            for i, (name, result) in enumerate(results.items()):
                if "y_prob" in result:
                    precision, recall, _ = precision_recall_curve(y_test, result["y_prob"])
                    avg_precision = result.get("avg_precision", average_precision_score(y_test, result["y_prob"]))
                    
                    # 使用不同的线型和颜色
                    linestyle = '-' if i < 5 else '--'
                    if name == best_model_name:
                        # 最佳模型使用较粗线条
                        plt.plot(recall, precision, linewidth=3, label=f'{name} (AP = {avg_precision:.4f})',
                                color=cmap(i), linestyle=linestyle)
                    else:
                        plt.plot(recall, precision, linewidth=2, label=f'{name} (AP = {avg_precision:.4f})',
                                color=cmap(i), linestyle=linestyle)
            
            # 添加随机基准线
            no_skill = len(y_test[y_test==1]) / len(y_test)
            plt.plot([0, 1], [no_skill, no_skill], 'k--', label=f'随机猜测 (AP = {no_skill:.4f})')
            
            plt.xlabel('召回率', fontsize=14)
            plt.ylabel('精确率', fontsize=14)
            plt.title('精确率-召回率曲线比较', fontsize=16, fontweight='bold')
            plt.legend(loc='best', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.xlim([-0.01, 1.01])
            plt.ylim([-0.01, 1.01])
            
            plt.tight_layout()
            plt.savefig('precision_recall_curves.png', dpi=300, bbox_inches='tight')
            print("\n精确率-召回率曲线已保存为: precision_recall_curves.png")
        except Exception as e:
            print(f"绘制精确率-召回率曲线时出错: {str(e)}")
        
        # 【新增】绘制校准曲线，评估概率预测的校准性
        try:
            plt.figure(figsize=(14, 12))
            
            # 绘制校准曲线
            ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
            ax2 = plt.subplot2grid((3, 1), (2, 0))
            
            ax1.plot([0, 1], [0, 1], "k:", label="理想校准")
            
            # 为每个模型绘制校准曲线
            for i, (name, model) in enumerate(best_models.items()):
                if hasattr(model, 'predict_proba'):
                    try:
                        # 计算并绘制校准曲线
                        disp = CalibrationDisplay.from_predictions(
                            y_test,
                            results[name]["y_prob"],
                            n_bins=10,
                            name=name,
                            ax=ax1,
                            color=cmap(i)
                        )
                        
                        # 绘制直方图
                        ax2.hist(
                            results[name]["y_prob"],
                            range=(0, 1),
                            bins=10,
                            label=name,
                            histtype="step",
                            lw=2,
                            color=cmap(i)
                        )
                    except:
                        print(f"绘制{name}校准曲线时出错")
            
            ax1.set_xlabel("预测概率", fontsize=14)
            ax1.set_ylabel("实际占比", fontsize=14)
            ax1.set_title("校准曲线 (预测概率校准性)", fontsize=16)
            ax1.grid(alpha=0.5)
            ax1.legend(loc="best")
            
            ax2.set_xlabel("平均预测概率", fontsize=14)
            ax2.set_ylabel("样本数", fontsize=14)
            ax2.set_title("预测概率分布", fontsize=16)
            ax2.grid(alpha=0.5)
            ax2.legend(loc="upper center", ncol=2)
            
            plt.tight_layout()
            plt.savefig('calibration_curves.png', dpi=300, bbox_inches='tight')
            print("\n校准曲线已保存为: calibration_curves.png")
        except Exception as e:
            print(f"绘制校准曲线时出错: {str(e)}")
        
        # 绘制特征重要性(对于最佳模型)
        try:
            if hasattr(best_model, 'feature_importances_') or best_model_name in ["随机森林", "XGBoost", "CatBoost", "LightGBM"]:
                # 获取特征重要性
                feature_importance = None
                
                if hasattr(best_model, 'steps') and len(best_model.steps) > 0:  # 检查是否是管道
                    model_key = best_model.steps[-1][0]
                    if hasattr(best_model.named_steps[model_key], 'feature_importances_'):
                        feature_importance = best_model.named_steps[model_key].feature_importances_
                elif hasattr(best_model, 'feature_importances_'):
                    feature_importance = best_model.feature_importances_
                
                if feature_importance is not None:
                    # 创建特征重要性DataFrame
                    importance_df = pd.DataFrame({
                        '特征': selected_features,
                        '重要性': feature_importance
                    }).sort_values('重要性', ascending=False)
                    
                    # 绘制特征重要性图
                    plt.figure(figsize=(14, 10))
                    
                    # 只显示前15个最重要的特征
                    top_features = importance_df.head(15)
                    
                    # 使用更美观的颜色地图
                    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(top_features)))
                    
                    # 绘制水平条形图
                    bars = plt.barh(top_features['特征'], top_features['重要性'], color=colors)
                    
                    # 添加数值标签
                    for i, bar in enumerate(bars):
                        plt.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2, 
                                f'{top_features["重要性"].iloc[i]:.4f}', 
                                va='center', fontsize=10)
                    
                    plt.title(f'{best_model_name}模型 - 特征重要性', fontsize=14, fontweight='bold')
                    plt.xlabel('重要性得分', fontsize=12)
                    plt.ylabel('特征', fontsize=12)
                    plt.grid(axis='x', linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    
                    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
                    print("\n特征重要性图已保存为: feature_importance.png")
                    
                    # 输出特征重要性
                    print("\n特征重要性排名:")
                    for i, (feature, importance) in enumerate(zip(importance_df['特征'].head(15), 
                                                                 importance_df['重要性'].head(15)), 1):
                        print(f"{i}. {feature}: {importance:.4f}")
        except Exception as e:
            print(f"计算特征重要性时出错: {str(e)}")
        
        # 绘制混淆矩阵热力图
        try:
            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(y_test, results[best_model_name]["y_pred"])
            
            # 使用更美观的颜色地图
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
            
            # 添加百分比标注
            group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
            group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
            labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
            labels = np.asarray(labels).reshape(2, 2)
            
            for i in range(2):
                for j in range(2):
                    plt.text(j+0.5, i+0.5, labels[i, j], 
                            ha="center", va="center", fontsize=12)
            
            plt.xlabel('预测标签', fontsize=12)
            plt.ylabel('真实标签', fontsize=12)
            plt.title(f'{best_model_name} - 混淆矩阵', fontsize=14, fontweight='bold')
            
            # 添加类别标签
            plt.xticks([0.5, 1.5], ['首次住院(0)', '多次住院(1)'], fontsize=10)
            plt.yticks([0.5, 1.5], ['首次住院(0)', '多次住院(1)'], fontsize=10, rotation=90)
            
            plt.tight_layout()
            plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
            print("\n混淆矩阵已保存为: confusion_matrix.png")
            
            # 计算并打印更详细的评估指标
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # 召回率/敏感性
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # 特异性
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # 阳性预测值
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # 阴性预测值
            
            print(f"\n详细评估指标:")
            print(f"敏感性(真阳性率): {sensitivity:.4f}")
            print(f"特异性(真阴性率): {specificity:.4f}")
            print(f"阳性预测值: {ppv:.4f}")
            print(f"阴性预测值: {npv:.4f}")
            
        except Exception as e:
            print(f"绘制混淆矩阵时出错: {str(e)}")
        
        # 绘制模型性能对比柱状图
        try:
            plt.figure(figsize=(16, 10))
            
            # 要比较的指标
            metrics = ['准确率', 'F1得分', 'AUC', '精确率', '召回率', '平均精确率']
            
            # 设置不同模型的颜色
            num_models = len(metrics_df)
            cmap = plt.cm.get_cmap('tab10', num_models)
            colors = [cmap(i) for i in range(num_models)]
            
            # 设置分组柱状图的位置
            x = np.arange(len(metrics))
            width = 0.8 / num_models  # 根据模型数量调整宽度
            offsets = np.linspace(-(num_models-1)/2*width, (num_models-1)/2*width, num_models)
            
            # 绘制每个模型的性能指标
            for i, (model_name, color, offset) in enumerate(zip(metrics_df['模型'], colors, offsets)):
                values = [
                    metrics_df.loc[metrics_df['模型'] == model_name, '准确率'].values[0],
                    metrics_df.loc[metrics_df['模型'] == model_name, 'F1得分'].values[0],
                    metrics_df.loc[metrics_df['模型'] == model_name, 'AUC'].values[0],
                    metrics_df.loc[metrics_df['模型'] == model_name, '精确率'].values[0],
                    metrics_df.loc[metrics_df['模型'] == model_name, '召回率'].values[0],
                    metrics_df.loc[metrics_df['模型'] == model_name, '平均精确率'].values[0]
                ]
                
                # 使用不同样式突出显示最佳模型
                if model_name == best_model_name:
                    # 为最佳模型添加边框
                    plt.bar(x + offset, values, width * 0.9, 
                            label=model_name, color=color, alpha=0.8,
                            edgecolor='black', linewidth=1.5)
                else:
                    plt.bar(x + offset, values, width * 0.9, 
                            label=model_name, color=color, alpha=0.7)
            
            # 添加标签和标题
            plt.xlabel('评估指标', fontsize=14)
            plt.ylabel('得分', fontsize=14)
            plt.title('不同模型在各评估指标上的表现对比', fontsize=16, fontweight='bold')
            plt.xticks(x, metrics, fontsize=12)
            plt.yticks(fontsize=12)
            
            # 调整图例
            plt.legend(title='模型', title_fontsize=13,
                    fontsize=11, loc='upper center', 
                    bbox_to_anchor=(0.5, -0.05), 
                    ncol=min(4, num_models),
                    frameon=True, fancybox=True, shadow=True)
            
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            
            # 限制y轴范围在0-1
            plt.ylim([0, 1.05])
            
            # 添加水平辅助线
            for y in [0.6, 0.7, 0.8, 0.9, 1.0]:
                plt.axhline(y=y, linestyle='--', alpha=0.3, color='gray')
            
            plt.tight_layout()
            plt.savefig('model_comparison_bar.png', dpi=300, bbox_inches='tight')
            print("\n模型性能对比图已保存为: model_comparison_bar.png")
            
        except Exception as e:
            print(f"绘制模型性能对比图时出错: {str(e)}")
        
        return best_model_name, best_model
    
    except Exception as e:
        print(f"比较模型时出错: {str(e)}")
        # 如果出错，返回任意一个可用模型
        if best_models and len(best_models) > 0:
            model_name = list(best_models.keys())[0]
            return model_name, best_models[model_name]
        return None, None

# 6. 模型保存与加载
def save_model(model, selected_features, file_name='lead_poisoning_model.pkl'):
    """保存模型和选定的特征"""
    import pickle
    
    # 保存模型和特征名称
    model_data = {
        'model': model,
        'selected_features': selected_features
    }
    
    with open(file_name, 'wb') as f:
        pickle.dump(model_data, f)
        
    print(f"\n模型和特征已保存至: {file_name}")

# 新增: 嵌套交叉验证函数
def nested_cross_validation(X, y, selected_features, outer_cv=5, inner_cv=3):
    """实现嵌套交叉验证评估模型泛化能力"""
    print("\n执行嵌套交叉验证评估模型泛化能力...")
    
    # 使用已选择的特征
    X = X[selected_features]
    
    # 检查样本量，调整交叉验证折数
    if len(X) < 50:
        outer_cv = 3
        inner_cv = 2
        print(f"  样本量较小({len(X)}个样本)，调整为{outer_cv}折外层交叉验证和{inner_cv}折内层交叉验证")
    else:
        print(f"  使用{outer_cv}折外层交叉验证和{inner_cv}折内层交叉验证")
    
    # 定义模型和参数空间
    models = {
        "CatBoost": {
            "model": CatBoostClassifier(verbose=0, random_seed=42),
            "params": {
                'iterations': Integer(50, 200),
                'depth': Integer(3, 7),
                'learning_rate': Real(0.01, 0.2),
                'l2_leaf_reg': Real(1, 5)
            }
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                'n_estimators': Integer(50, 200),
                'max_depth': Integer(3, 10),
                'min_samples_split': Integer(2, 8)
            }
        },
        "XGBoost": {
            "model": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            "params": {
                'n_estimators': Integer(50, 200),
                'max_depth': Integer(2, 6),
                'learning_rate': Real(0.01, 0.2)
            }
        }
    }
    
    # 设置外层交叉验证
    outer_cv_splitter = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=42)
    
    # 存储结果
    nested_scores = {model_name: [] for model_name in models.keys()}
    feature_importances = {model_name: pd.DataFrame() for model_name in models.keys()}
    
    # 外层交叉验证
    fold = 1
    for train_idx, test_idx in outer_cv_splitter.split(X, y):
        print(f"\n  外层交叉验证 - 第{fold}/{outer_cv}折")
        fold += 1
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # 在训练集上标准化数据
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # 内层交叉验证用于超参数调优
        for model_name, model_info in models.items():
            print(f"    训练和优化{model_name}...")
            inner_cv_splitter = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=42)
            
            # 贝叶斯优化超参数
            optimizer = BayesSearchCV(
                estimator=model_info["model"],
                search_spaces=model_info["params"],
                n_iter=20,
                cv=inner_cv_splitter,
                scoring='roc_auc',
                n_jobs=1,
                random_state=42
            )
            
            # 在训练集上拟合最佳模型
            optimizer.fit(X_train_scaled, y_train)
            best_model = optimizer.best_estimator_
            
            # 在测试集上评估
            y_pred = best_model.predict(X_test_scaled)
            y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
            
            # 存储性能指标
            score = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'auc': roc_auc_score(y_test, y_prob)
            }
            nested_scores[model_name].append(score)
            
            # 打印当前折的结果
            print(f"      AUC: {score['auc']:.4f}, F1: {score['f1']:.4f}")
            
            # 保存特征重要性（如果模型支持）
            if hasattr(best_model, 'feature_importances_'):
                fold_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': best_model.feature_importances_
                })
                feature_importances[model_name] = pd.concat([feature_importances[model_name], fold_importance])
    
    # 汇总结果
    final_scores = {}
    for model_name, scores in nested_scores.items():
        avg_scores = {metric: np.mean([s[metric] for s in scores]) for metric in scores[0].keys()}
        std_scores = {metric: np.std([s[metric] for s in scores]) for metric in scores[0].keys()}
        final_scores[model_name] = {'avg': avg_scores, 'std': std_scores}
    
    # 汇总特征重要性
    final_importances = {}
    for model_name, importances in feature_importances.items():
        if not importances.empty:
            final_importances[model_name] = importances.groupby('feature')['importance'].mean().sort_values(ascending=False)
    
    return final_scores, final_importances

# 首先添加全局的explain_prediction函数
def explain_prediction(patient_data, model, selected_features, importances):
    """为单个患者生成临床解释"""
    # 准备数据
    if isinstance(patient_data, pd.DataFrame):
        X_patient = patient_data[selected_features]
    else:
        X_patient = pd.DataFrame([patient_data], columns=selected_features)
        
    # 预测风险
    risk_proba = model.predict_proba(X_patient)[0, 1]
    risk_class = "多次住院风险" if risk_proba > 0.5 else "首次住院"
    
    # 找出最重要的特征及其值
    top_features = importances.head(5)['特征'].tolist()
    patient_values = {feature: X_patient[feature].values[0] for feature in top_features}
    
    # 生成解释
    explanation = {
        '预测结果': risk_class,
        '复发风险概率': f"{risk_proba:.2%}",
        '主要影响因素': {
            feature: {
                '值': patient_values[feature],
                '重要性': importances.loc[importances['特征'] == feature, '评分权重'].values[0]
            } for feature in top_features
        },
        '临床建议': "需密切随访，考虑加强解毒治疗" if risk_proba > 0.7 else 
                  ("适当加强监测，调整治疗方案" if risk_proba > 0.5 else "常规随访")
    }
    
    return explanation

# 新增: 模型可解释性增强函数
def enhance_interpretability(model, X, selected_features, test_size=0.25, random_state=42):
    """增强模型可解释性，创建临床风险评分系统"""
    print("\n增强模型可解释性...")
    
    # 划分训练和评估集
    X_train, X_eval = train_test_split(X, test_size=test_size, random_state=random_state)
    
    # 获取特征重要性
    feature_importances = None
    
    # 尝试从模型获取特征重要性
    try:
        if hasattr(model, 'get_feature_importances'):
            # 使用集成模型自定义方法
            feature_importances = model.get_feature_importances()
        elif hasattr(model, 'feature_importances_'):
            # 对于RandomForest, XGBoost等
            feature_importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # 对于线性模型
            feature_importances = np.abs(model.coef_[0])
        elif hasattr(model, 'best_estimator_') and hasattr(model.best_estimator_, 'feature_importances_'):
            # 对于GridSearchCV等包装器
            feature_importances = model.best_estimator_.feature_importances_
        elif hasattr(model, 'named_steps'):
            # 对于Pipeline
            for _, step in model.named_steps.items():
                if hasattr(step, 'feature_importances_'):
                    feature_importances = step.feature_importances_
                    break
                elif hasattr(step, 'coef_'):
                    feature_importances = np.abs(step.coef_[0])
                    break
    except Exception as e:
        print(f"获取特征重要性时出错: {str(e)}")
    
    # 如果无法获取内置特征重要性，尝试使用SHAP值
    if feature_importances is None or len(feature_importances) != len(selected_features):
        try:
            print("使用SHAP值作为特征重要性的替代")
            # 使用小样本计算SHAP值
            sample_size = min(100, X.shape[0])
            X_sample = X.sample(sample_size, random_state=random_state) if isinstance(X, pd.DataFrame) else X[:sample_size]
            
            if not SHAP_AVAILABLE:
                print("SHAP库不可用，无法生成模型解释")
                return None
            
            try:
                explainer = shap.KernelExplainer(
                    lambda x: model.predict_proba(x)[:, 1] if hasattr(model, 'predict_proba') else model.predict(x),
                    X_sample
                )
                shap_values = explainer.shap_values(X_sample)
                
                # 对于多输出模型，取最后一个输出
                if isinstance(shap_values, list):
                    shap_values = shap_values[-1]
                
                # 计算平均绝对SHAP值作为特征重要性
                feature_importances = np.mean(np.abs(shap_values), axis=0)
            except Exception as e:
                print(f"SHAP解释器创建失败: {str(e)}")
                return None
        except:
            print("无法创建完整的风险评分系统，缺少特征重要性信息")
            return None
    
    # 创建特征重要性数据框
    if feature_importances is not None and len(feature_importances) == len(selected_features):
        importances = pd.DataFrame({
            '特征': selected_features,
            '重要性': feature_importances
        })
        importances = importances.sort_values('重要性', ascending=False)
        
        # 创建临床风险评分系统
        # 将特征重要性标准化为0-10分
        max_importance = importances['重要性'].max()
        if max_importance > 0:
            importances['风险分数'] = (importances['重要性'] / max_importance * 10).round(1)
            
            # 打印风险评分系统
            print("\n临床风险评分系统:")
            for _, row in importances.head(15).iterrows():
                print(f"  {row['特征']}: {row['风险分数']} 分")
                
            # 创建风险评分可视化
            plt.figure(figsize=(10, 8))
            plt.barh(importances.head(15)['特征'], importances.head(15)['风险分数'], color='skyblue')
            plt.xlabel('风险分数 (0-10)')
            plt.ylabel('临床特征')
            plt.title('铅中毒复发风险评分系统')
            plt.tight_layout()
            plt.savefig('lead_poisoning_risk_score.png', dpi=300, bbox_inches='tight')
            print("风险评分系统图已保存为: lead_poisoning_risk_score.png")
            
            return importances
    
    print("该模型不支持内置特征重要性，无法创建完整的风险评分系统")
    return None

# 新增: 保存优化后的模型和解释系统
def save_optimized_model(model, selected_features, interpretability_system, file_name='lead_poisoning_optimized_model.pkl'):
    """保存模型、特征和解释系统"""
    import pickle
    
    # 保存模型、特征名称和解释系统
    model_data = {
        'model': model,
        'selected_features': selected_features,
        'interpretability_system': interpretability_system
    }
    
    try:
        with open(file_name, 'wb') as f:
            pickle.dump(model_data, f)
            
        print(f"\n模型、特征和解释系统已保存至: {file_name}")
    except Exception as e:
        print(f"保存模型时出错: {str(e)}")
        # 尝试保存不包含解释系统的简化版本
        try:
            simple_model_data = {
                'model': model,
                'selected_features': selected_features
            }
            with open(file_name, 'wb') as f:
                pickle.dump(simple_model_data, f)
            print(f"已保存简化版模型(不含解释系统)至: {file_name}")
        except Exception as e2:
            print(f"保存简化版模型也失败: {str(e2)}")

# 新增: 高级采样策略函数
def apply_advanced_sampling(X, y, sampling_strategy='auto', method='combined', random_state=42):
    """
    应用高级采样技术处理类别不平衡问题
    
    参数:
    - X: 特征矩阵
    - y: 目标变量
    - sampling_strategy: 采样策略，默认'auto'根据数据自动确定
    - method: 采样方法，可选'over'(过采样), 'under'(欠采样), 'combined'(混合采样)
    - random_state: 随机种子
    
    返回:
    - X_resampled, y_resampled: 重采样后的数据集
    """
    if not IMBALANCED_AVAILABLE:
        print("imbalanced-learn库不可用，将返回原始数据")
        return X, y
    
    # 检查数据不平衡程度
    class_counts = np.bincount(y)
    imbalance_ratio = class_counts[0] / class_counts[1] if class_counts[1] > 0 else float('inf')
    
    print(f"应用高级采样技术处理不平衡数据 (比例 {imbalance_ratio:.2f}:1)")
    
    # 根据不平衡程度和数据集大小选择合适的方法
    small_dataset = len(X) < 100
    severe_imbalance = imbalance_ratio > 3.0
    
    try:
        # 针对小数据集的特殊处理
        if small_dataset:
            if method == 'over' or method == 'combined':
                # 小数据集使用BorderlineSMOTE更保守
                print("小数据集使用BorderlineSMOTE过采样...")
                sampler = BorderlineSMOTE(
                    sampling_strategy=sampling_strategy,
                    k_neighbors=min(5, int(class_counts[1] / 2)) if class_counts[1] > 2 else 1,
                    random_state=random_state
                )
            else:
                # 小数据集使用基本的随机欠采样
                print("小数据集使用RandomUnderSampler欠采样...")
                sampler = RandomUnderSampler(
                    sampling_strategy=sampling_strategy,
                    random_state=random_state
                )
        else:
            # 根据方法选择合适的采样器
            if method == 'over':
                # 严重不平衡使用ADASYN，否则使用SMOTE
                if severe_imbalance:
                    print("使用ADASYN自适应合成过采样...")
                    sampler = ADASYN(
                        sampling_strategy=sampling_strategy,
                        n_neighbors=min(5, int(class_counts[1] / 2)) if class_counts[1] > 2 else 1,
                        random_state=random_state
                    )
                else:
                    print("使用SMOTE过采样...")
                    sampler = SMOTE(
                        sampling_strategy=sampling_strategy,
                        k_neighbors=min(5, int(class_counts[1] / 2)) if class_counts[1] > 2 else 1,
                        random_state=random_state
                    )
            elif method == 'under':
                # 数据量足够使用聚类欠采样，否则使用随机欠采样
                if len(X) > 200:
                    print("使用ClusterCentroids聚类欠采样...")
                    sampler = ClusterCentroids(
                        sampling_strategy=sampling_strategy,
                        random_state=random_state
                    )
                else:
                    print("使用RandomUnderSampler随机欠采样...")
                    sampler = RandomUnderSampler(
                        sampling_strategy=sampling_strategy,
                        random_state=random_state
                    )
            else:  # 'combined'
                # 混合采样方法，结合SMOTE与Tomek Links清理
                print("使用SMOTETomek混合采样...")
                sampler = SMOTETomek(
                    sampling_strategy=sampling_strategy,
                    random_state=random_state
                )
                
        # 执行采样
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        # 输出采样结果
        new_class_counts = np.bincount(y_resampled)
        print(f"采样前: 类别0: {class_counts[0]}个样本, 类别1: {class_counts[1]}个样本")
        print(f"采样后: 类别0: {new_class_counts[0]}个样本, 类别1: {new_class_counts[1]}个样本")
        
        return X_resampled, y_resampled
        
    except Exception as e:
        print(f"应用高级采样技术时出错: {str(e)}")
        print("返回原始数据集")
        return X, y

def main():
    """主函数"""
    print("=== 铅中毒预测模型 ===")
    print("目标: 预测患者是否为多次住院(铅中毒复发风险)")
    
    # 加载数据
    df = load_data('aldata.xlsx')
    
    # 特征工程和数据清洗
    X, y, num_cols, cat_cols = preprocess_data(df)
    
    # 检查预处理是否成功
    if X is None or y is None:
        print("数据预处理失败，无法继续建模")
        return
    
    # 特征选择
    X_selected, selected_features = select_features(X, y, num_cols)
    
    # 模型构建与评估
    results, best_models, X_test, y_test = build_models(X_selected, y)
    
    # 模型比较与可视化
    best_model_name, best_model = compare_models(results, best_models, X_test, y_test, selected_features)
    
    # 保存最佳模型
    save_model(best_model, selected_features)
    
    print("\n=== 建模过程完成 ===")
    print(f"最佳模型: {best_model_name}")
    print(f"选中特征数量: {len(selected_features)}")
    print("模型已保存，可用于新数据预测")

# 新增: 优化版主函数
def optimized_main():
    """优化后的主函数，集成高级特征工程、采样技术、融合模型和可解释性增强"""
    print("=== 铅中毒预测模型优化版 ===")
    print("目标: 预测患者是否为多次住院(铅中毒复发风险)")
    
    # 加载数据
    df = load_data('aldata.xlsx')
    
    # 基础特征工程和数据清洗
    X, y, num_cols, cat_cols = preprocess_data(df)
    
    # 检查预处理是否成功
    if X is None or y is None:
        print("数据预处理失败，无法继续建模")
        return
    
    # 高级特征工程 - 增强版
    X = advanced_feature_engineering(X)
    
    # 特征选择
    X_selected, selected_features = select_features(X, y, num_cols)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # 应用高级采样技术处理训练数据
    X_train_resampled, y_train_resampled = apply_advanced_sampling(
        X_train, y_train, 
        method='combined', 
        random_state=42
    )
    
    # 训练测试集标准化
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_resampled), 
        columns=X_train_resampled.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )
    
    print("\n执行嵌套交叉验证评估模型泛化能力...")
    final_scores, final_importances = nested_cross_validation(X_selected, y, selected_features)
    
    # 展示嵌套交叉验证结果
    print("\n嵌套交叉验证结果 (更可靠的泛化性能估计):")
    for model_name, scores in final_scores.items():
        print(f"\n{model_name}:")
        print(f"  平均AUC: {scores['avg']['auc']:.4f} ± {scores['std']['auc']:.4f}")
        print(f"  平均F1: {scores['avg']['f1']:.4f} ± {scores['std']['f1']:.4f}")
        print(f"  平均准确率: {scores['avg']['accuracy']:.4f} ± {scores['std']['accuracy']:.4f}")
        print(f"  平均精确率: {scores['avg']['precision']:.4f} ± {scores['std']['precision']:.4f}")
        print(f"  平均召回率: {scores['avg']['recall']:.4f} ± {scores['std']['recall']:.4f}")
    
    # 确定嵌套验证的最佳模型
    best_cv_model_name = max(final_scores, key=lambda k: final_scores[k]['avg']['auc'])
    print(f"\n嵌套验证最佳模型: {best_cv_model_name}")
    
    # 创建基础模型列表
    base_models = []
    
    # 添加常规模型
    base_models.append((
        'LogisticRegression', 
        LogisticRegression(C=1.0, max_iter=2000, random_state=42)
    ))
    
    base_models.append((
        'RandomForest',
        RandomForestClassifier(n_estimators=150, max_depth=5, random_state=42)
    ))
    
    base_models.append((
        'SVC',
        SVC(C=1.0, kernel='rbf', probability=True, random_state=42)
    ))
    
    # 有条件地添加其他模型
    if XGBOOST_AVAILABLE:
        base_models.append((
            'XGBoost',
            XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05,
                         random_state=42, use_label_encoder=False, eval_metric='logloss')
        ))
    
    if CATBOOST_AVAILABLE:
        base_models.append((
            'CatBoost',
            CatBoostClassifier(iterations=150, depth=5, learning_rate=0.05,
                              verbose=0, random_seed=42)
        ))
    
    if LIGHTGBM_AVAILABLE:
        base_models.append((
            'LightGBM',
            LGBMClassifier(
                n_estimators=150, 
                max_depth=5, 
                learning_rate=0.05,
                min_child_samples=5,  # 增加最小子节点样本数
                min_child_weight=1,   # 设置最小子节点权重
                min_split_gain=0.01,  # 设置最小分裂增益
                subsample=0.8,        # 子采样比例
                colsample_bytree=0.8, # 特征采样比例
                reg_alpha=0.1,        # L1正则化
                reg_lambda=1.0,       # L2正则化
                random_state=42,
                verbose=-1            # 禁用冗余警告
            )
        ))
    
    # 创建元模型 - 使用简单的逻辑回归
    meta_model = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
    
    # 创建优化的集成模型
    ensemble_model = OptimizedEnsembleClassifier(
        base_models=base_models,
        meta_model=meta_model,
        use_probas=True,
        optimize_weights=True,
        calibrate_probas=True,
        random_state=42
    )
    
    # 训练集成模型
    print("\n训练高级集成模型...")
    try:
        ensemble_model.fit(X_train_scaled, y_train_resampled)
    except Exception as e:
        print(f"训练集成模型失败: {str(e)}")
        print("使用备选策略创建集成模型...")
        # 创建简化版的集成模型（VotingClassifier）
        ensemble_model = VotingClassifier(
            estimators=base_models,
            voting='soft',
            n_jobs=1
        )
        ensemble_model.fit(X_train_scaled, y_train_resampled)
    
    # 在测试集上评估集成模型
    print("\n评估集成模型...")
    y_pred_ensemble = ensemble_model.predict(X_test_scaled)
    y_prob_ensemble = ensemble_model.predict_proba(X_test_scaled)[:, 1]
    
    # 计算性能指标
    accuracy = accuracy_score(y_test, y_pred_ensemble)
    precision = precision_score(y_test, y_pred_ensemble, zero_division=0)
    recall = recall_score(y_test, y_pred_ensemble, zero_division=0)
    f1 = f1_score(y_test, y_pred_ensemble, zero_division=0)
    auc_score = roc_auc_score(y_test, y_prob_ensemble)
    
    print("\n集成模型评估结果:")
    print(f"  准确率: {accuracy:.4f}")
    print(f"  精确率: {precision:.4f}")
    print(f"  召回率: {recall:.4f}")
    print(f"  F1得分: {f1:.4f}")
    print(f"  AUC: {auc_score:.4f}")
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred_ensemble)
    tn, fp, fn, tp = cm.ravel()
    
    print("\n混淆矩阵:")
    print(cm)
    
    print(f"  真阳性: {tp}, 假阳性: {fp}")
    print(f"  假阴性: {fn}, 真阴性: {tn}")
    
    # 应用临床阈值优化
    print("\n优化临床决策阈值...")
    
    # 设置漏诊（假阴性）的代价比误诊（假阳性）高
    cost_weights = {'fn_cost': 5, 'fp_cost': 1}  # 漏诊代价是误诊的5倍
    
    optimal_threshold, threshold_metrics = optimize_clinical_threshold(
        y_test, y_prob_ensemble, cost_weights=cost_weights
    )
    
    # 使用最优阈值重新预测
    y_pred_optimal = (y_prob_ensemble >= optimal_threshold).astype(int)
    
    # 计算优化后的性能指标
    accuracy_opt = accuracy_score(y_test, y_pred_optimal)
    precision_opt = precision_score(y_test, y_pred_optimal, zero_division=0)
    recall_opt = recall_score(y_test, y_pred_optimal, zero_division=0)
    f1_opt = f1_score(y_test, y_pred_optimal, zero_division=0)
    
    print(f"\n使用最优临床阈值 ({optimal_threshold:.4f}) 的性能:")
    print(f"  准确率: {accuracy_opt:.4f}")
    print(f"  精确率: {precision_opt:.4f}")
    print(f"  召回率: {recall_opt:.4f}")
    print(f"  F1得分: {f1_opt:.4f}")
    
    # 使用SHAP增强模型可解释性
    print("\n生成高级模型解释...")
    
    # 生成SHAP解释
    shap_values, shap_plots = None, []
    if SHAP_AVAILABLE:
        shap_values, shap_plots = generate_shap_explanations(
            ensemble_model, X_test_scaled, 
            feature_names=selected_features,
            plot_type='dependence',
            n_samples=min(100, X_test_scaled.shape[0])
        )
    
    # 如果SHAP不可用，使用传统的特征重要性
    interpretability_results = enhance_interpretability(ensemble_model, X_selected, selected_features)
    
    # 保存模型和解释系统
    model_data = {
        'model': ensemble_model,
        'selected_features': selected_features,
        'optimal_threshold': optimal_threshold,
        'interpretability_system': interpretability_results,
        'scaler': scaler
    }
    
    try:
        import pickle
        with open('lead_poisoning_advanced_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        print("\n高级模型已保存至: lead_poisoning_advanced_model.pkl")
    except Exception as e:
        print(f"保存模型时出错: {str(e)}")
    
    # 在原始未采样数据上训练单独的最佳模型作为备份
    if best_cv_model_name == "CatBoost" and CATBOOST_AVAILABLE:
        backup_model = CatBoostClassifier(iterations=150, depth=5, learning_rate=0.05, verbose=0, random_seed=42)
    elif best_cv_model_name == "RandomForest":
        backup_model = RandomForestClassifier(n_estimators=150, max_depth=5, random_state=42)
    elif best_cv_model_name == "XGBoost" and XGBOOST_AVAILABLE:
        backup_model = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, 
                                   random_state=42, use_label_encoder=False, eval_metric='logloss')
    else:
        # 默认使用随机森林作为备份
        backup_model = RandomForestClassifier(n_estimators=150, max_depth=5, random_state=42)
    
    print(f"\n在全部数据上训练备份模型 ({type(backup_model).__name__})...")
    backup_model.fit(X_selected, y)
    save_model(backup_model, selected_features, 'lead_poisoning_backup_model.pkl')
    
    print("\n=== 高级优化模型构建完成 ===")
    print(f"集成模型类型: {type(ensemble_model).__name__}")
    print(f"最优临床阈值: {optimal_threshold:.4f}")
    print(f"选中特征数量: {len(selected_features)}")
    print("模型已保存，可用于新数据预测和个体风险评估")
    
    return ensemble_model, selected_features, optimal_threshold

# 新增: 优化的模型融合系统
class OptimizedEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    优化的集成分类器，结合多种基础模型，并实现自适应权重调整
    """
    def __init__(self, base_models=None, meta_model=None, use_probas=True, 
                 optimize_weights=True, calibrate_probas=True, random_state=42):
        """
        初始化集成分类器
        
        参数:
        - base_models: 基础模型列表，每个元素是(名称, 模型)元组
        - meta_model: 元学习器(二级模型)
        - use_probas: 是否使用预测概率作为元特征
        - optimize_weights: 是否优化模型权重
        - calibrate_probas: 是否校准预测概率
        - random_state: 随机种子
        """
        self.base_models = base_models if base_models is not None else []
        self.meta_model = meta_model
        self.use_probas = use_probas
        self.optimize_weights = optimize_weights
        self.calibrate_probas = calibrate_probas
        self.random_state = random_state
        self.models_ = []
        self.meta_model_ = None
        self.weights_ = None
        self.calibrators_ = []
        
    def fit(self, X, y):
        """训练集成分类器"""
        if self.base_models is None or len(self.base_models) == 0:
            raise ValueError("必须提供至少一个基础模型")
            
        # 检查是否需要分割校准数据集
        if self.calibrate_probas:
            # 预留20%的数据用于校准
            X_train, X_cal, y_train, y_cal = train_test_split(
                X, y, test_size=0.2, random_state=self.random_state, stratify=y
            )
        else:
            X_train, X_cal, y_train, y_cal = X, X, y, y
            
        # 保存特征名称(如果X是DataFrame)
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            
        # 训练所有基础模型
        self.fitted_models_ = {}
        for name, model in self.base_models:
            if model is not None:
                try:
                    print(f"训练基础模型: {name}")
                    model_clone = clone(model)
                    model_clone.fit(X_train, y_train)
                    self.fitted_models_[name] = model_clone
                except Exception as e:
                    print(f"训练{name}模型时出错: {str(e)}")
                    self.fitted_models_[name] = None
                    
        # 校准模型概率
        if self.calibrate_probas and len(self.fitted_models_) > 0:
            try:
                print("校准模型概率...")
                self.calibrated_models_ = self._calibrate_probas(self.fitted_models_, X_cal, y_cal)
                # 替换原始模型为校准后的模型
                for name, model in self.calibrated_models_.items():
                    if model is not None:
                        self.fitted_models_[name] = model
            except Exception as e:
                print(f"校准模型概率时出错: {str(e)}")
                # 继续使用未校准模型
                self.calibrated_models_ = None
                
        # 如果需要优化权重，使用校准数据集评估各个模型并分配权重
        if self.optimize_weights:
            try:
                print("优化模型权重...")
                self.weights_ = {}
                
                # 对每个模型计算AUC
                for name, model in self.fitted_models_.items():
                    if model is not None:
                        try:
                            # 获取模型在校准数据集上的性能
                            if hasattr(model, 'predict_proba'):
                                y_prob = model.predict_proba(X_cal)[:,1]
                                auc = roc_auc_score(y_cal, y_prob)
                                # 使用AUC作为权重，缩放到[0-1]并提高非线性差异
                                weight = np.clip(auc ** 2, 0.5, 1.0)
                                self.weights_[name] = weight
                                print(f"  {name} 权重: {weight:.4f} (AUC: {auc:.4f})")
                            else:
                                self.weights_[name] = 0.5
                        except Exception as e:
                            print(f"计算{name}模型AUC时出错: {str(e)}")
                            self.weights_[name] = 0.5
                            
            except Exception as e:
                print(f"优化模型权重时出错: {str(e)}")
                self.weights_ = None
                
        # 如果没有设置权重优化或者优化失败，使用相等权重
        if not hasattr(self, 'weights_') or self.weights_ is None:
            self.weights_ = {name: 1.0 for name, _ in self.fitted_models_.items() if _ is not None}
            
        # 训练元模型（如果有）
        if self.meta_model is not None:
            try:
                print("训练元模型...")
                
                # 为元模型生成特征（来自基础模型的预测）
                meta_features = self.get_meta_features(X_cal, self.fitted_models_)
                
                # 训练元模型
                self.meta_model.fit(meta_features, y_cal)
                
                # 添加到fitted_models中
                self.fitted_models_['meta'] = self.meta_model
            except Exception as e:
                print(f"训练元模型时出错: {str(e)}")
                self.meta_model = None
                
        return self
    
    def optimize_model_weights(self, X_meta, y_meta):
        """优化模型权重"""
        # 使用各模型在验证集上的AUC作为权重
        weights = []
        for i, (name, _) in enumerate(self.models_):
            try:
                auc_score = roc_auc_score(y_meta, X_meta[:, i])
                # 应用非线性变换增强差异
                weight = max(0.1, auc_score ** 2)
                weights.append(weight)
                print(f"  {name} 权重: {weight:.4f} (AUC: {auc_score:.4f})")
            except:
                # 如果出错使用默认权重
                weights.append(1.0)
                print(f"  {name} 权重: 1.0000 (默认)")
        
        # 归一化权重
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        
        self.weights_ = weights
    
    # 添加获取元特征的方法
    def get_meta_features(self, X, base_models):
        """为元模型生成特征，从基础模型的预测中"""
        meta_features = []
        
        for name, model in base_models.items():
            if model is not None:
                try:
                    if hasattr(model, 'predict_proba'):
                        # 使用预测概率作为特征
                        proba = model.predict_proba(X)[:, 1].reshape(-1, 1)
                        meta_features.append(proba)
                    else:
                        # 对于不支持概率的模型，使用预测类别
                        pred = model.predict(X).reshape(-1, 1)
                        meta_features.append(pred)
                except Exception as e:
                    print(f"获取{name}模型元特征时出错: {str(e)}")
                    # 出错时使用0.5填充
                    meta_features.append(np.ones((X.shape[0], 1)) * 0.5)
        
        # 水平合并所有特征
        if meta_features:
            return np.hstack(meta_features)
        else:
            # 如果没有有效的特征，返回全0特征
            return np.zeros((X.shape[0], 1))

    def _calibrate_probas(self, base_models, X_cal, y_cal):
        """校准各个模型的预测概率"""
        calibrated_models = {}
        for name, model in base_models.items():
            if model is not None:
                try:
                    # 确保使用有效模型
                    if hasattr(model, 'predict_proba'):
                        # 创建校准器
                        calibrator = CalibratedClassifierCV(model, cv='prefit')
                        calibrator.fit(X_cal, y_cal)
                        calibrated_models[name] = calibrator
                    else:
                        print(f"模型 {name} 不支持概率预测，跳过校准")
                        calibrated_models[name] = model
                except Exception as e:
                    print(f"校准{name}模型概率时出错: {str(e)}")
                    calibrated_models[name] = model
            else:
                print(f"模型 {name} 为空，跳过校准")
                
        return calibrated_models

    # 修复预测部分，确保即使校准失败也能工作
    def predict(self, X):
        """预测类别"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X):
        """预测概率"""
        # 获取各个基础模型的预测概率
        base_probas = {}
        for name, model in self.fitted_models_.items():
            try:
                if model is not None:
                    if hasattr(model, 'predict_proba'):
                        base_probas[name] = model.predict_proba(X)[:, 1].reshape(-1, 1)
                    elif hasattr(model, 'decision_function'):
                        # 对于SVM等使用决策函数
                        decision = model.decision_function(X).reshape(-1, 1)
                        base_probas[name] = 1 / (1 + np.exp(-decision))
            except Exception as e:
                print(f"获取{name}模型预测概率时出错: {str(e)}")
                # 出错时使用0.5作为默认值
                base_probas[name] = np.ones((X.shape[0], 1)) * 0.5
                
        # 如果没有任何可用的预测，返回默认概率
        if not base_probas:
            return np.ones((X.shape[0], 2)) * 0.5
        
        # 应用模型权重
        if self.weights_ is not None:
            weighted_probas = []
            for name, proba in base_probas.items():
                if name in self.weights_:
                    weighted_probas.append(proba * self.weights_[name])
                else:
                    weighted_probas.append(proba)
            
            # 计算加权平均
            if weighted_probas:
                avg_proba = np.sum(weighted_probas, axis=0) / len(weighted_probas)
                # 确保概率范围在[0,1]
                avg_proba = np.clip(avg_proba, 0, 1)
                # 返回二分类概率 [1-p, p]
                return np.hstack([1-avg_proba, avg_proba])
            
        # 如果没有权重或出错，返回平均概率
        avg_proba = np.mean([proba for proba in base_probas.values()], axis=0)
        avg_proba = np.clip(avg_proba, 0, 1)
        return np.hstack([1-avg_proba, avg_proba])
        
    def get_feature_importances(self):
        """获取特征重要性"""
        if not hasattr(self, 'fitted_models_'):
            return None
            
        importances = {}
        for name, model in self.fitted_models_.items():
            # 尝试从不同模型中获取特征重要性
            try:
                if model is None:
                    continue
                    
                if hasattr(model, 'feature_importances_'):
                    # RandomForest, XGBoost等
                    importances[name] = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    # 线性模型
                    importances[name] = np.abs(model.coef_[0])
                elif hasattr(model, 'feature_importance_'):
                    # CatBoost
                    importances[name] = model.feature_importance()
                elif hasattr(model, 'feature_importance'):
                    # LightGBM
                    importances[name] = model.feature_importance()
            except Exception as e:
                print(f"获取{name}特征重要性时出错: {str(e)}")
                
        if not importances:
            return None
            
        # 计算加权平均特征重要性
        models_with_importances = len(importances)
        if models_with_importances > 0:
            # 获取所有特征重要性的维度
            first_model = list(importances.keys())[0]
            n_features = len(importances[first_model])
            
            # 初始化平均特征重要性
            avg_importance = np.zeros(n_features)
            
            # 计算平均值
            for name, imp in importances.items():
                if len(imp) == n_features:  # 确保维度匹配
                    # 标准化特征重要性
                    normalized_imp = imp / np.sum(imp) if np.sum(imp) > 0 else imp
                    avg_importance += normalized_imp
                    
            # 最终平均值
            avg_importance /= models_with_importances
            return avg_importance
        
        return None

# 辅助函数: 安全克隆模型
def clone_model_safely(model):
    """安全地克隆模型，处理不同库的模型"""
    try:
        from sklearn.base import clone
        return clone(model)
    except:
        # 对于不支持sklearn clone的模型，尝试直接复制参数
        try:
            model_type = type(model)
            params = model.get_params()
            return model_type(**params)
        except:
            # 最后的尝试：创建同类型新实例
            try:
                return type(model)()
            except:
                # 如果都失败了，返回原始模型（风险：可能会修改原始模型）
                print(f"警告: 无法克隆模型 {type(model).__name__}，使用原始模型")
                return model

# 新增: 创建优化的集成模型
def create_optimized_ensemble(base_models=None, meta_algorithm='stacking', calibrate=True, random_state=42):
    """
    创建优化的集成模型
    
    参数:
    - base_models: 基础模型列表，如果为None则自动创建
    - meta_algorithm: 'stacking'或'voting'
    - calibrate: 是否校准预测概率
    - random_state: 随机种子
    
    返回:
    - ensemble_model: 优化的集成模型
    """
    print("\n创建优化的集成模型...")
    
    # 如果没有提供基础模型，创建默认模型
    if base_models is None:
        base_models = []
        
        # 添加逻辑回归
        base_models.append(('LogisticRegression', LogisticRegression(
            C=1.0, max_iter=2000, random_state=random_state
        )))
        
        # 添加随机森林
        base_models.append(('RandomForest', RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=random_state
        )))
        
        # 添加SVM
        base_models.append(('SVM', SVC(
            probability=True, C=1.0, kernel='rbf', random_state=random_state
        )))
        
        # 有条件地添加其他模型
        if XGBOOST_AVAILABLE:
            base_models.append(('XGBoost', XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1, 
                use_label_encoder=False, eval_metric='logloss',
                random_state=random_state
            )))
        
        if CATBOOST_AVAILABLE:
            base_models.append(('CatBoost', CatBoostClassifier(
                iterations=100, depth=5, learning_rate=0.1,
                verbose=0, random_state=random_state
            )))
        
        if LIGHTGBM_AVAILABLE:
            base_models.append(('LightGBM', LGBMClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=random_state
            )))
    
    # 根据选择的元算法创建集成模型
    if meta_algorithm.lower() == 'stacking':
        print("使用Stacking作为元算法")
        # 创建Stacking集成
        ensemble_model = OptimizedEnsembleClassifier(
            base_models=base_models,
            meta_model=LogisticRegression(C=1.0, random_state=random_state),
            use_probas=True,
            optimize_weights=True,
            calibrate_probas=calibrate,
            random_state=random_state
        )
    else:  # voting
        print("使用Voting作为元算法")
        # 创建sklearn的VotingClassifier
        ensemble_model = VotingClassifier(
            estimators=base_models,
            voting='soft',  # 使用预测概率
            n_jobs=1
        )
    
    return ensemble_model

# 新增: 临床阈值优化函数
def optimize_clinical_threshold(y_true, y_prob, cost_weights=None):
    """
    优化分类阈值，考虑临床成本
    
    参数:
    - y_true: 真实标签
    - y_prob: 预测概率(正类)
    - cost_weights: 代价权重字典，格式为 {'fn_cost': 5, 'fp_cost': 1}
                   fn_cost是漏诊成本，fp_cost是误诊成本
    
    返回:
    - optimal_threshold: 最优阈值
    - metrics: 各阈值的性能指标字典
    """
    print("优化临床分类阈值...")
    
    # 如果没有提供成本权重，使用默认值
    if cost_weights is None:
        # 默认漏诊(假阴性)比误诊(假阳性)代价更高
        cost_weights = {'fn_cost': 5, 'fp_cost': 1}
    
    print(f"代价设置: 漏诊代价={cost_weights['fn_cost']}, 误诊代价={cost_weights['fp_cost']}")
    
    # 初始化变量
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = []
    metrics = {
        'thresholds': thresholds,
        'sensitivity': [],
        'specificity': [],
        'f1': [],
        'precision': [],
        'npv': [],
        'accuracy': [],
        'cost': []
    }
    
    # 计算每个阈值的性能指标和代价
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        
        # 计算混淆矩阵
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # 计算性能指标
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        f1 = 2 * precision_val * sensitivity / (precision_val + sensitivity) if (precision_val + sensitivity) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        # 计算代价: 考虑假阳性和假阴性的不同成本
        cost = cost_weights['fp_cost'] * fp + cost_weights['fn_cost'] * fn
        
        # 存储指标
        metrics['sensitivity'].append(sensitivity)
        metrics['specificity'].append(specificity)
        metrics['precision'].append(precision_val)
        metrics['npv'].append(npv)
        metrics['f1'].append(f1)
        metrics['accuracy'].append(accuracy)
        metrics['cost'].append(cost)
        
        costs.append(cost)
    
    # 找到最小代价对应的阈值
    optimal_idx = np.argmin(costs)
    optimal_threshold = thresholds[optimal_idx]
    
    # 计算最优阈值下的指标
    print(f"最优阈值: {optimal_threshold:.4f}")
    print(f"敏感性: {metrics['sensitivity'][optimal_idx]:.4f}")
    print(f"特异性: {metrics['specificity'][optimal_idx]:.4f}")
    print(f"F1分数: {metrics['f1'][optimal_idx]:.4f}")
    
    # 绘制ROC曲线和最优阈值点
    try:
        plt.figure(figsize=(10, 8))
        
        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # 绘制ROC曲线
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机猜测')
        
        # 找出最优阈值对应的点
        optimal_threshold_idx = np.abs(np.array(metrics['thresholds']) - optimal_threshold).argmin()
        opt_sens = metrics['sensitivity'][optimal_threshold_idx]
        opt_spec = metrics['specificity'][optimal_threshold_idx]
        opt_fp_rate = 1 - opt_spec
        
        # 在ROC曲线上标出最优阈值点
        plt.scatter(opt_fp_rate, opt_sens, marker='o', color='red', s=100, 
                   label=f'最优阈值: {optimal_threshold:.4f}')
        
        plt.xlabel('假阳性率 (1 - 特异性)')
        plt.ylabel('真阳性率 (敏感性)')
        plt.title('ROC曲线与最优临床阈值')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        # 保存图像
        plt.savefig('optimal_clinical_threshold.png', dpi=300, bbox_inches='tight')
        print("最优临床阈值图已保存为: optimal_clinical_threshold.png")
    except Exception as e:
        print(f"绘制最优阈值图时出错: {str(e)}")
    
    return optimal_threshold, metrics

# 新增: SHAP解释函数
def generate_shap_explanations(model, X, feature_names=None, plot_type='summary', n_samples=100):
    """
    使用SHAP生成模型解释
    
    参数:
    - model: 训练好的模型
    - X: 特征矩阵
    - feature_names: 特征名称列表
    - plot_type: 'summary'或'dependence'
    - n_samples: 用于计算SHAP值的样本数量
    
    返回:
    - shap_values: SHAP值
    - shap_plots: 已保存的SHAP图表路径列表
    """
    if not SHAP_AVAILABLE:
        print("SHAP库不可用，无法生成SHAP解释")
        return None, []
    
    print("生成SHAP模型解释...")
    
    try:
        # 确定要使用的样本数
        if X.shape[0] > n_samples:
            print(f"使用{n_samples}个样本计算SHAP值 (总样本数: {X.shape[0]})")
            X_sample = X.sample(n_samples, random_state=42)
        else:
            print(f"使用全部{X.shape[0]}个样本计算SHAP值")
            X_sample = X
        
        # 选择合适的SHAP解释器
        # 首先尝试基于树的解释器（更快）
        if hasattr(model, 'feature_importances_'):
            print("使用基于树的SHAP解释器")
            explainer = shap.TreeExplainer(model)
        else:
            # 退回到KernelExplainer（更慢但更通用）
            print("使用核SHAP解释器")
            # 定义预测函数
            if hasattr(model, 'predict_proba'):
                def predict_fn(x):
                    return model.predict_proba(x)[:, 1]
            else:
                def predict_fn(x):
                    return model.predict(x)
                
            # 创建解释器
            explainer = shap.KernelExplainer(predict_fn, shap.kmeans(X_sample, 10))
        
        # 计算SHAP值
        shap_values = explainer.shap_values(X_sample)
        
        # 如果返回的是一个列表（多类别的情况），取第一个元素
        if isinstance(shap_values, list) and len(shap_values) > 0:
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        # 检查特征名称
        if feature_names is None:
            if isinstance(X, pd.DataFrame):
                feature_names = X.columns.tolist()
            else:
                feature_names = [f"特征{i}" for i in range(X.shape[1])]
        
        # 保存的图表路径列表
        saved_plots = []
        
        # 绘制SHAP摘要图
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.title("SHAP特征重要性摘要", fontsize=14)
        plt.tight_layout()
        plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        saved_plots.append('shap_summary.png')
        print("SHAP摘要图已保存为: shap_summary.png")
        
        # 绘制SHAP条形图
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type='bar', show=False)
        plt.title("SHAP特征重要性排序", fontsize=14)
        plt.tight_layout()
        plt.savefig('shap_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        saved_plots.append('shap_importance.png')
        print("SHAP重要性图已保存为: shap_importance.png")
        
        # 绘制前3个最重要特征的依赖图
        if plot_type == 'dependence' and isinstance(X, pd.DataFrame):
            # 获取最重要的特征索引
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            most_important = np.argsort(mean_abs_shap)[-3:]
            
            for idx in most_important:
                feature_name = feature_names[idx]
                plt.figure(figsize=(10, 7))
                shap.dependence_plot(idx, shap_values, X_sample, feature_names=feature_names, show=False)
                plt.title(f"SHAP依赖图: {feature_name}", fontsize=14)
                plt.tight_layout()
                
                file_name = f'shap_dependence_{feature_name}.png'
                safe_name = ''.join([c if c.isalnum() else '_' for c in file_name])
                plt.savefig(safe_name, dpi=300, bbox_inches='tight')
                plt.close()
                saved_plots.append(safe_name)
                print(f"SHAP依赖图已保存为: {safe_name}")
        
        return shap_values, saved_plots
        
    except Exception as e:
        print(f"生成SHAP解释时出错: {str(e)}")
        return None, []

# 修改现有函数，移除集成模型部分
def run_fixed_model():
    """运行修复后的模型，比较单个机器学习模型的性能并选择最好的一个"""
    print("=== 运行修复后的铅中毒预测模型 ===")
    
    # 加载数据
    df = load_data('aldata.xlsx')
    
    # 基础特征工程和数据清洗
    X, y, num_cols, cat_cols = preprocess_data(df)
    
    # 检查预处理是否成功
    if X is None or y is None:
        print("数据预处理失败，无法继续建模")
        return
    
    # 高级特征工程
    X = advanced_feature_engineering(X)
    
    # 特征选择
    X_selected, selected_features = select_features(X, y, num_cols)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # 标准化数据
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), 
        columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns
    )
    
    # 创建单个模型字典
    models = {}
    
    # 添加常规模型
    models["逻辑回归"] = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
    models["随机森林"] = RandomForestClassifier(n_estimators=150, max_depth=5, random_state=42)
    models["SVC"] = SVC(C=1.0, kernel='rbf', probability=True, random_state=42)
    
    # 有条件地添加其他模型
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=150, 
            max_depth=4, 
            learning_rate=0.05,
            random_state=42, 
            use_label_encoder=False, 
            eval_metric='logloss'
        )
    
    if CATBOOST_AVAILABLE:
        models["CatBoost"] = CatBoostClassifier(
            iterations=150, 
            depth=5, 
            learning_rate=0.05,
            verbose=0, 
            random_seed=42
        )
    
    if LIGHTGBM_AVAILABLE:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=150, 
            max_depth=5, 
            learning_rate=0.05,
            min_child_samples=5,  
            min_child_weight=1,   
            min_split_gain=0.01,  
            subsample=0.8,        
            colsample_bytree=0.8, 
            reg_alpha=0.1,        
            reg_lambda=1.0,       
            random_state=42,
            verbose=-1            
        )
    
    # 存储每个模型的性能指标
    results = {}
    
    # 训练和评估每个模型
    print("\n训练和评估各个模型...")
    for name, model in models.items():
        print(f"\n训练模型: {name}")
        
        # 训练模型
        model.fit(X_train_scaled, y_train)
        
        # 在测试集上评估
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # 计算性能指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc_score = roc_auc_score(y_test, y_prob)
        
        # 存储结果
        results[name] = {
            "model": model,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc_score,
            "y_pred": y_pred,
            "y_prob": y_prob
        }
        
        print(f"模型评估结果:")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  精确率: {precision:.4f}")
        print(f"  召回率: {recall:.4f}")
        print(f"  F1得分: {f1:.4f}")
        print(f"  AUC: {auc_score:.4f}")
        
        # 显示混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        print("\n混淆矩阵:")
        print(cm)
    
    # 比较模型性能，选择最佳模型
    print("\n比较各模型性能:")
    models_df = pd.DataFrame({
        "模型": list(results.keys()),
        "准确率": [results[m]["accuracy"] for m in results],
        "精确率": [results[m]["precision"] for m in results],
        "召回率": [results[m]["recall"] for m in results],
        "F1得分": [results[m]["f1"] for m in results],
        "AUC": [results[m]["auc"] for m in results]
    }).sort_values("AUC", ascending=False)
    
    print(models_df)
    
    # 选择最佳模型 (基于AUC)
    best_model_name = models_df.iloc[0]["模型"]
    best_model = results[best_model_name]["model"]
    best_y_prob = results[best_model_name]["y_prob"]
    
    print(f"\n最佳模型: {best_model_name}")
    print(f"AUC: {results[best_model_name]['auc']:.4f}")
    
    # 应用临床阈值优化
    print("\n优化临床决策阈值...")
    
    # 设置漏诊（假阴性）的代价比误诊（假阳性）高
    cost_weights = {'fn_cost': 5, 'fp_cost': 1}  # 漏诊代价是误诊的5倍
    
    optimal_threshold, threshold_metrics = optimize_clinical_threshold(
        y_test, best_y_prob, cost_weights=cost_weights
    )
    
    # 使用最优阈值重新预测
    y_pred_optimal = (best_y_prob >= optimal_threshold).astype(int)
    
    # 计算优化后的性能指标
    accuracy_opt = accuracy_score(y_test, y_pred_optimal)
    precision_opt = precision_score(y_test, y_pred_optimal, zero_division=0)
    recall_opt = recall_score(y_test, y_pred_optimal, zero_division=0)
    f1_opt = f1_score(y_test, y_pred_optimal, zero_division=0)
    
    print(f"\n使用最优临床阈值 ({optimal_threshold:.4f}) 的性能:")
    print(f"  准确率: {accuracy_opt:.4f}")
    print(f"  精确率: {precision_opt:.4f}")
    print(f"  召回率: {recall_opt:.4f}")
    print(f"  F1得分: {f1_opt:.4f}")
    
    # 保存最佳模型
    print(f"\n保存最佳模型: {best_model_name}")
    model_data = {
        'model': best_model,
        'selected_features': selected_features,
        'optimal_threshold': optimal_threshold,
        'scaler': scaler
    }
    
    try:
        import pickle
        with open('lead_poisoning_best_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        print(f"最佳模型 ({best_model_name}) 已保存至: lead_poisoning_best_model.pkl")
    except Exception as e:
        print(f"保存模型时出错: {str(e)}")
    
    print("\n=== 模型训练和评估完成 ===")
    return best_model, selected_features, optimal_threshold

# 如果直接运行此脚本，则执行修复的模型
if __name__ == "__main__":
    run_fixed_model() 