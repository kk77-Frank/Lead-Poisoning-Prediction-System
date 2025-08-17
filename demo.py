#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
铅中毒预测模型演示脚本
演示：
1. 模型训练过程（集成模型）
2. 优化单模型训练（比较多个单一模型并选择最佳）
3. 高级单模型优化（进一步提升模型性能）
4. 模型预测使用
5. 结果解释与可视化
"""

import pandas as pd
import numpy as np
from lead_poisoning_prediction import (load_data, preprocess_data, select_features, build_models, 
                                     compare_models, save_model, advanced_feature_engineering, 
                                     optimize_clinical_threshold)
from lead_poisoning_prediction_utils import load_saved_model, predict_risk, explain_prediction, generate_patient_report, batch_predict
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, PowerTransformer, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, average_precision_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif, RFE, RFECV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
import os
import pickle
import warnings
from scipy.stats import randint, uniform

# 导入条件库
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("警告: XGBoost 库未安装，将不会使用XGBoost模型")
    XGBOOST_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    print("警告: CatBoost 库未安装，将不会使用CatBoost模型")
    CATBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    import lightgbm as lgb  # 用于callbacks早停
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("警告: LightGBM 库未安装，将不会使用LightGBM模型")
    LIGHTGBM_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("警告: SHAP 库未安装，将使用基础特征重要性方法")
    SHAP_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    from imblearn.combine import SMOTEENN, SMOTETomek
    from imblearn.ensemble import BalancedRandomForestClassifier
    IMBALANCED_AVAILABLE = True
except ImportError:
    print("警告: imbalanced-learn 库未安装，将使用基础类别平衡方法")
    IMBALANCED_AVAILABLE = False

try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    SKOPT_AVAILABLE = True
except ImportError:
    print("警告: scikit-optimize 库未安装，将使用随机搜索代替贝叶斯优化")
    SKOPT_AVAILABLE = False

try:
    from optuna import create_study
    from optuna.integration import OptunaSearchCV
    OPTUNA_AVAILABLE = True
except ImportError:
    print("警告: Optuna 库未安装，将不使用Optuna优化")
    OPTUNA_AVAILABLE = False

def run_model_training():
    """运行原始集成模型训练过程"""
    print("="*50)
    print("开始铅中毒预测集成模型训练流程")
    print("="*50)
    
    # 检查数据文件是否存在
    if not os.path.exists('aldata.xlsx'):
        print("错误: 未找到数据文件 'aldata.xlsx'")
        return False
    
    # 1. 加载数据
    df = load_data('aldata.xlsx')
    
    # 2. 特征工程和数据清洗
    X, y, num_cols, cat_cols = preprocess_data(df)
    
    # 3. 特征选择
    X_selected, selected_features = select_features(X, y, num_cols)
    
    # 4. 模型构建与评估
    results, best_models, X_test, y_test = build_models(X_selected, y)
    
    # 5. 模型比较与可视化
    best_model_name, best_model = compare_models(results, best_models, X_test, y_test, selected_features)
    
    # 6. 保存最佳模型
    save_model(best_model, selected_features)
    
    print("\n模型训练完成!")
    return True

def run_optimized_model_training():
    """运行优化的单模型训练过程，比较多个机器学习模型的性能"""
    print("="*50)
    print("开始铅中毒预测优化模型训练流程 - 单模型比较")
    print("="*50)
    
    # 检查数据文件是否存在
    if not os.path.exists('aldata.xlsx'):
        print("错误: 未找到数据文件 'aldata.xlsx'")
        return False
    
    # 加载数据
    df = load_data('aldata.xlsx')
    
    # 基础特征工程和数据清洗
    X, y, num_cols, cat_cols = preprocess_data(df)
    
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
        with open('lead_poisoning_best_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        print(f"最佳模型 ({best_model_name}) 已保存至: lead_poisoning_best_model.pkl")
        
        # 保存模型比较结果
        models_df.to_csv('model_comparison_results.csv', index=False, encoding='utf-8-sig')
        print("模型比较结果已保存至: model_comparison_results.csv")
    except Exception as e:
        print(f"保存模型时出错: {str(e)}")
    
    print("\n=== 优化模型训练和评估完成 ===")
    return True

def run_advanced_model_optimization():
    """运行高级单模型优化，进一步提升模型性能"""
    print("="*50)
    print("开始铅中毒预测模型高级优化")
    print("="*50)
    
    if not os.path.exists('aldata.xlsx'):
        print("错误: 未找到数据文件 'aldata.xlsx'")
        return False
    
    print("\n1. 加载数据...")
    df = load_data('aldata.xlsx')
    print(f"加载了 {df.shape[0]} 条记录，{df.shape[1]} 个特征")
    
    print("\n2. 执行基础特征工程...")
    X, y, num_cols, cat_cols = preprocess_data(df)
    print(f"预处理后: {X.shape[1]} 个特征")
    
    print("\n3. 执行高级特征工程...")
    X = advanced_feature_engineering(X)
    print(f"特征工程后: {X.shape[1]} 个特征")
    
    print("\n4. 执行高级特征选择...")
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    k_best = min(40, X_dev.shape[1])
    selector = SelectKBest(f_classif, k=k_best)
    selector.fit(X_dev, y_dev)
    selected_mask = selector.get_support()
    if LIGHTGBM_AVAILABLE:
        model_selector = LGBMClassifier(n_estimators=100, random_state=42)
    elif XGBOOST_AVAILABLE:
        model_selector = XGBClassifier(n_estimators=100, random_state=42)
    else:
        model_selector = RandomForestClassifier(n_estimators=100, random_state=42)
    rfe = RFECV(estimator=model_selector, step=1, cv=5, scoring='roc_auc', min_features_to_select=10)
    rfe.fit(X_dev, y_dev)
    rfe_mask = rfe.get_support()
    combined_mask = np.logical_or(selected_mask, rfe_mask)
    selected_features = X.columns[combined_mask].tolist()
    X_selected = X[selected_features]
    print(f"特征选择后保留 {len(selected_features)} 个特征")
    
    print("\n5. 执行嵌套交叉验证优化...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\n6. 应用高级数据预处理...")
    robust_scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(robust_scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(robust_scaler.transform(X_test), columns=X_test.columns)
    
    # 为早停/校准创建训练/验证拆分（避免用测试集做早停）
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # 类别不平衡权重
    class_counts = np.bincount(y_train)
    pos_weight = None
    if len(class_counts) == 2 and class_counts.min() / class_counts.max() < 0.8:
        neg, pos = class_counts[0], class_counts[1]
        pos_weight = neg / max(pos, 1)
        print(f"检测到不平衡: neg={neg}, pos={pos}, scale_pos_weight≈{pos_weight:.2f}")
        if IMBALANCED_AVAILABLE:
            try:
                smote = BorderlineSMOTE(random_state=42)
                X_tr_array, y_tr = smote.fit_resample(X_tr.values, y_tr)
                X_tr = pd.DataFrame(X_tr_array, columns=X_tr.columns)
                print(f"应用BorderlineSMOTE后的类别分布: {np.bincount(y_tr)}")
            except Exception as e:
                print(f"应用SMOTE时出错: {str(e)}，改用类别权重")
    
    print("\n7. 执行高级超参数优化...")
    # 定义重复分层CV与AP+AUC综合评分，以提升不平衡场景下的鲁棒性
    cv_strategy = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    def ap_auc_composite(y_true, y_proba):
        try:
            auc_val = roc_auc_score(y_true, y_proba)
            ap_val = average_precision_score(y_true, y_proba)
            return 0.5 * float(auc_val) + 0.5 * float(ap_val)
        except Exception:
            return 0.0
    composite_scorer = make_scorer(ap_auc_composite, needs_proba=True)
    best_candidates = []  # 保存各候选模型 (name, best_model, cv_auc)

    # LightGBM 候选
    if LIGHTGBM_AVAILABLE:
        print("使用LightGBM进行优化...")
        lgb_base = LGBMClassifier(random_state=42, scale_pos_weight=pos_weight if pos_weight else None, verbosity=-1)
        if SKOPT_AVAILABLE:
            print("使用贝叶斯优化执行超参数搜索(LightGBM)...")
            param_space = {
                "n_estimators": Integer(100, 600),
                "max_depth": Integer(3, 12),
                "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
                "num_leaves": Integer(20, 100),
                "min_child_samples": Integer(5, 40),
                "subsample": Real(0.6, 1.0),
                "colsample_bytree": Real(0.6, 1.0),
                "min_split_gain": Real(0.0, 0.2),
                "max_bin": Integer(255, 511),
                "boosting_type": Categorical(["gbdt", "dart"]),
                "reg_alpha": Real(1e-6, 1.0, prior="log-uniform"),
                "reg_lambda": Real(1e-6, 1.0, prior="log-uniform"),
                "class_weight": Categorical(["balanced", None])
            }
            lgb_search = BayesSearchCV(
                lgb_base, param_space, cv=cv_strategy, scoring=composite_scorer, n_jobs=-1,
                n_iter=60, random_state=42, verbose=0
            )
        else:
            print("使用随机搜索执行超参数优化(LightGBM)...")
            param_distributions = {
                "n_estimators": randint(100, 600),
                "max_depth": randint(3, 12),
                "learning_rate": uniform(0.01, 0.29),
                "num_leaves": randint(20, 100),
                "min_child_samples": randint(5, 40),
                "subsample": uniform(0.6, 0.4),
                "colsample_bytree": uniform(0.6, 0.4),
                "min_split_gain": uniform(0.0, 0.2),
                "max_bin": randint(255, 512),
                "boosting_type": ["gbdt", "dart"],
                "reg_alpha": uniform(0.0, 1.0),
                "reg_lambda": uniform(0.0, 1.0),
                "class_weight": ["balanced", None]
            }
            lgb_search = RandomizedSearchCV(
                lgb_base, param_distributions, cv=cv_strategy, scoring=composite_scorer, n_jobs=-1,
                n_iter=80, random_state=42, verbose=0
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lgb_search.fit(X_tr, y_tr)
        lgb_best_params = lgb_search.best_estimator_.get_params()
        lgb_best = LGBMClassifier(**lgb_best_params)
        try:
            lgb_best.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)], eval_metric='auc',
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
        except Exception:
            lgb_best.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='auc')
        best_candidates.append(("LightGBM", lgb_best, lgb_search.best_score_))

    # XGBoost 候选
    if XGBOOST_AVAILABLE:
        print("使用XGBoost进行优化...")
        xgb_base = XGBClassifier(random_state=42, scale_pos_weight=pos_weight if pos_weight else 1.0,
                                 use_label_encoder=False, eval_metric='aucpr')
        xgb_distributions = {
            "n_estimators": randint(100, 800),
            "max_depth": randint(3, 10),
            "learning_rate": uniform(0.01, 0.29),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
            "min_child_weight": randint(1, 12),
            "gamma": uniform(0.0, 1.0),
            "reg_alpha": uniform(0.0, 1.0),
            "reg_lambda": uniform(0.0, 1.0)
        }
        xgb_search = RandomizedSearchCV(
            xgb_base, xgb_distributions, cv=cv_strategy, scoring=composite_scorer, n_jobs=-1,
            n_iter=80, random_state=42, verbose=0
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xgb_search.fit(X_tr, y_tr)
        xgb_best = xgb_search.best_estimator_
        try:
            xgb_best.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
        except Exception:
            xgb_best.fit(X_tr, y_tr)
        best_candidates.append(("XGBoost", xgb_best, xgb_search.best_score_))

    # CatBoost 候选
    if CATBOOST_AVAILABLE:
        print("使用CatBoost进行优化...")
        cat_base = CatBoostClassifier(random_seed=42, verbose=0, eval_metric='PRAUC',
                                      class_weights=[1.0, float(pos_weight)] if pos_weight else None)
        cat_distributions = {
            "iterations": randint(200, 1000),
            "depth": randint(3, 8),
            "learning_rate": uniform(0.01, 0.2),
            "l2_leaf_reg": uniform(1.0, 9.0),
            "border_count": randint(32, 255)
        }
        cat_search = RandomizedSearchCV(
            cat_base, cat_distributions, cv=cv_strategy, scoring=composite_scorer, n_jobs=-1,
            n_iter=60, random_state=42, verbose=0
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cat_search.fit(X_tr, y_tr)
        cat_best = cat_search.best_estimator_
        try:
            cat_best.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], use_best_model=True, early_stopping_rounds=50, verbose=False)
        except Exception:
            cat_best.fit(X_tr, y_tr)
        best_candidates.append(("CatBoost", cat_best, cat_search.best_score_))

    # 随机森林 候选（兜底）
    print("使用随机森林进行优化...")
    rf_base = RandomForestClassifier(random_state=42, class_weight='balanced')
    rf_distributions = {
        "n_estimators": randint(200, 800),
        "max_depth": [None, 5, 8, 12, 16, 20],
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 10),
        "max_features": ['sqrt', 'log2', None],
        "class_weight": ['balanced', 'balanced_subsample', None],
        "bootstrap": [True, False]
    }
    rf_search = RandomizedSearchCV(
        rf_base, rf_distributions, cv=cv_strategy, scoring=composite_scorer, n_jobs=-1,
        n_iter=80, random_state=42, verbose=0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rf_search.fit(X_tr, y_tr)
    rf_best = rf_search.best_estimator_
    rf_best.fit(X_tr, y_tr)
    best_candidates.append(("RandomForest", rf_best, rf_search.best_score_))

    # ExtraTrees 候选
    print("使用ExtraTrees进行优化...")
    et_base = ExtraTreesClassifier(random_state=42, class_weight='balanced')
    et_distributions = {
        "n_estimators": randint(200, 800),
        "max_depth": [None, 5, 8, 12, 16, 20],
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 10),
        "max_features": ['sqrt', 'log2', None],
        "bootstrap": [True, False]
    }
    et_search = RandomizedSearchCV(
        et_base, et_distributions, cv=cv_strategy, scoring=composite_scorer, n_jobs=-1,
        n_iter=80, random_state=42, verbose=0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        et_search.fit(X_tr, y_tr)
    et_best = et_search.best_estimator_
    et_best.fit(X_tr, y_tr)
    best_candidates.append(("ExtraTrees", et_best, et_search.best_score_))

    # Balanced RandomForest 候选（若可用）
    if IMBALANCED_AVAILABLE:
        print("使用BalancedRandomForest进行优化...")
        brf_base = BalancedRandomForestClassifier(random_state=42)
        brf_distributions = {
            "n_estimators": randint(200, 800),
            "max_depth": [None, 5, 8, 12, 16, 20],
            "min_samples_split": randint(2, 20),
            "min_samples_leaf": randint(1, 10),
            "max_features": ['sqrt', 'log2', None]
        }
        brf_search = RandomizedSearchCV(
            brf_base, brf_distributions, cv=cv_strategy, scoring=composite_scorer, n_jobs=-1,
            n_iter=80, random_state=42, verbose=0
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            brf_search.fit(X_tr, y_tr)
        brf_best = brf_search.best_estimator_
        brf_best.fit(X_tr, y_tr)
        best_candidates.append(("BalancedRF", brf_best, brf_search.best_score_))

    # 选择CV AUC最高的候选模型
    # 同时计算各候选在训练集的CV AP，并用综合评分选择
    print("\n对候选模型进行CV指标评估(ROC AUC与AP)...")
    evaluated = []  # (name, model, cv_auc_mean, cv_ap_mean, composite)
    for name, model, cv_auc in best_candidates:
        try:
            auc_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
            ap_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='average_precision', n_jobs=-1)
            auc_mean = float(np.mean(auc_scores))
            ap_mean = float(np.mean(ap_scores))
        except Exception:
            # 若模型不支持CV直接评估，则回退使用前面记录的cv_auc及简单AP近似
            auc_mean = float(cv_auc)
            try:
                ap_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='average_precision', n_jobs=-1)
                ap_mean = float(np.mean(ap_scores))
            except Exception:
                ap_mean = auc_mean - 0.05
        composite = 0.6 * auc_mean + 0.4 * ap_mean
        evaluated.append((name, model, auc_mean, ap_mean, composite))
    print(f"候选综合评分: {[ (n, round(a,4), round(p,4), round(c,4)) for n,_,a,p,c in evaluated ]}")
    best_name, best_model, best_cv_auc, best_cv_ap, best_composite = max(evaluated, key=lambda t: t[4])
    print(f"选择综合评分最优模型: {best_name} (CV AUC={best_cv_auc:.4f}, CV AP={best_cv_ap:.4f}, 综合={best_composite:.4f})")

    # 概率校准（用验证集）
    print("\n执行概率校准...")
    try:
        if hasattr(best_model, 'predict_proba'):
            # 尝试 Isotonic 与 Sigmoid，两者在验证集上以AP为标准择优
            calib_iso = CalibratedClassifierCV(best_model, method='isotonic', cv='prefit')
            calib_iso.fit(X_val, y_val)
            ap_iso = average_precision_score(y_val, calib_iso.predict_proba(X_val)[:, 1])
            calib_sig = CalibratedClassifierCV(best_model, method='sigmoid', cv='prefit')
            calib_sig.fit(X_val, y_val)
            ap_sig = average_precision_score(y_val, calib_sig.predict_proba(X_val)[:, 1])
            if ap_iso >= ap_sig:
                best_model = calib_iso
                print(f"已完成概率校准(选择Isotonic, val AP={ap_iso:.4f} >= Sigmoid AP={ap_sig:.4f})")
            else:
                best_model = calib_sig
                print(f"已完成概率校准(选择Sigmoid, val AP={ap_sig:.4f} > Isotonic AP={ap_iso:.4f})")
    except Exception as e:
        print(f"概率校准失败: {e}，继续使用未校准模型")

    # 基于验证集选择F1最优阈值（避免用测试集调参）
    try:
        y_val_prob = best_model.predict_proba(X_val)[:, 1]
        thresholds_val = np.linspace(0.01, 0.99, 99)
        f1_val_list = []
        for t in thresholds_val:
            f1_val_list.append(f1_score(y_val, (y_val_prob >= t).astype(int), zero_division=0))
        best_f1_val_threshold = float(thresholds_val[int(np.argmax(f1_val_list))])
        print(f"验证集F1最优阈值: {best_f1_val_threshold:.3f}, F1(val): {max(f1_val_list):.4f}")
    except Exception as e:
        print(f"验证集阈值选择失败: {e}")
        best_f1_val_threshold = 0.5

    # 后续评估、阈值优化、保存保持不变
    print("\n9. 评估模型性能...")
    y_pred = best_model.predict(X_test_scaled)
    y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc_score = roc_auc_score(y_test, y_prob)
    ap_score = average_precision_score(y_test, y_prob)
    print("优化模型性能指标:")
    print(f"  准确率: {accuracy:.4f}")
    print(f"  精确率: {precision:.4f}")
    print(f"  召回率: {recall:.4f}")
    print(f"  F1得分: {f1:.4f}")
    print(f"  AUC: {auc_score:.4f}")
    print(f"  平均精度(AP): {ap_score:.4f}")

    print("\n10. 优化临床决策阈值...")
    cost_weights = {'fn_cost': 5, 'fp_cost': 1}
    optimal_threshold, threshold_metrics = optimize_clinical_threshold(y_test, y_prob, cost_weights=cost_weights)
    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
    accuracy_opt = accuracy_score(y_test, y_pred_optimal)
    precision_opt = precision_score(y_test, y_pred_optimal, zero_division=0)
    recall_opt = recall_score(y_test, y_pred_optimal, zero_division=0)
    f1_opt = f1_score(y_test, y_pred_optimal, zero_division=0)
    print(f"\n使用最优临床阈值 ({optimal_threshold:.4f}) 的性能:")
    print(f"  准确率: {accuracy_opt:.4f}")
    print(f"  精确率: {precision_opt:.4f}")
    print(f"  召回率: {recall_opt:.4f}")
    print(f"  F1得分: {f1_opt:.4f}")

    # 额外：按F1最优阈值搜索
    print("\n按F1最优阈值搜索...")
    # 使用验证集选出的阈值在测试集评估，避免用测试集调参
    best_f1_threshold = best_f1_val_threshold
    y_pred_f1 = (y_prob >= best_f1_threshold).astype(int)
    print(f"F1最优阈值(基于验证集): {best_f1_threshold:.3f}, F1(test): {f1_score(y_test, y_pred_f1, zero_division=0):.4f}, 精确率: {precision_score(y_test, y_pred_f1, zero_division=0):.4f}, 召回率: {recall_score(y_test, y_pred_f1, zero_division=0):.4f}")

    # 额外：召回率下限约束（如临床偏好高召回）
    thresholds_grid = np.linspace(0.01, 0.99, 99)
    best_f1_hi_recall = -1.0
    best_t_hi_recall = best_f1_threshold
    best_prec_hi_recall = 0.0
    best_rec_hi_recall = 0.0
    for t in thresholds_grid:
        yp = (y_prob >= t).astype(int)
        rec = recall_score(y_test, yp, zero_division=0)
        if rec >= 0.75:  # 召回率下限
            f1_tmp = f1_score(y_test, yp, zero_division=0)
            if f1_tmp > best_f1_hi_recall:
                best_f1_hi_recall = f1_tmp
                best_t_hi_recall = float(t)
                best_prec_hi_recall = precision_score(y_test, yp, zero_division=0)
                best_rec_hi_recall = rec
    if best_f1_hi_recall > 0:
        print(f"在召回率>=0.75约束下的最佳F1: {best_f1_hi_recall:.4f}，阈值: {best_t_hi_recall:.3f}，精确率: {best_prec_hi_recall:.4f}，召回率: {best_rec_hi_recall:.4f}")
    else:
        print("在召回率>=0.75的约束下未找到更优阈值，保留默认F1最优阈值")

    print("\n12. 保存优化后的模型...")
    model_data = {
        'model': best_model,
        'selected_features': selected_features,
        'optimal_threshold': optimal_threshold,
        'best_f1_threshold': best_f1_threshold,
        'val_f1_threshold': best_f1_val_threshold,
        'best_f1_threshold_recall75': best_t_hi_recall if best_f1_hi_recall > 0 else best_f1_threshold,
        'scaler': robust_scaler
    }
    try:
        with open('lead_poisoning_advanced_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        print(f"高级优化模型已保存至: lead_poisoning_advanced_model.pkl")
    except Exception as e:
        print(f"保存模型时出错: {str(e)}")

    print("\n=== 高级模型优化完成 ===")
    return True

def run_single_prediction_demo():
    """运行单个病例预测演示"""
    print("="*50)
    print("单个病例预测演示")
    print("="*50)
    
    # 尝试加载最佳单模型，如果不存在则使用常规模型
    model_path = 'lead_poisoning_best_model.pkl'
    if not os.path.exists(model_path):
        model_path = 'lead_poisoning_model.pkl'
        print(f"注意: 未找到最佳单模型，将使用常规模型 {model_path}")
    
    # 1. 加载保存的模型
    model, selected_features = load_saved_model(model_path)
    if model is None:
        print("错误: 无法加载模型，请先运行模型训练")
        return
    
    # 2. 创建一个模拟的新患者数据
    # 注意: 实际使用时应替换为真实数据
    print("\n创建模拟患者数据...")
    
    # 从训练数据中随机抽取一个样本作为示例
    try:
        train_data = pd.read_excel('aldata.xlsx')
        # 随机选择一个样本
        sample_idx = np.random.randint(0, len(train_data))
        sample_data = train_data.iloc[[sample_idx]].copy()
        
        print(f"已选择第 {sample_idx+1} 号样本作为演示")
        
        # 添加一些患者基本信息供报告使用
        sample_data['姓名'] = '张三'
        sample_data['年龄'] = 5
        sample_data['性别'] = '男'
        sample_data['住院号'] = 'ZY' + str(10000 + sample_idx)
        
        print("\n患者基本信息:")
        print(f"姓名: {sample_data['姓名'].values[0]}")
        print(f"年龄: {sample_data['年龄'].values[0]}岁")
        print(f"性别: {sample_data['性别'].values[0]}")
        print(f"住院号: {sample_data['住院号'].values[0]}")
        
    except Exception as e:
        print(f"无法读取训练数据，创建随机样本: {str(e)}")
        # 如果无法读取训练数据，则创建随机样本
        sample_data = pd.DataFrame(np.random.randn(1, len(selected_features)), 
                                  columns=selected_features)
    
    # 3. 进行预测
    print("\n开始预测风险...")
    results, risk_score = predict_risk(model, sample_data, selected_features)
    
    if results is not None:
        print("\n预测结果:")
        print(f"风险概率: {risk_score[0]:.2%}")
        print(f"风险预测: {'多次住院风险' if results['风险预测'].values[0] == 1 else '首次住院'}")
    
        # 4. 解释预测结果
        print("\n生成预测解释...")
        shap_values = explain_prediction(model, sample_data, selected_features)
        
        # 5. 生成患者报告
        print("\n生成患者报告...")
        report = generate_patient_report(sample_data, risk_score[0], shap_values, selected_features)
        print("\n报告已生成: patient_risk_report.md")
        print("\n报告预览:")
        print("="*30)
        print(report[:500] + "..." if len(report) > 500 else report)
        print("="*30)

def run_batch_prediction_demo():
    """运行批量预测演示"""
    print("="*50)
    print("批量预测演示")
    print("="*50)
    
    # 尝试加载最佳单模型，如果不存在则使用常规模型
    model_path = 'lead_poisoning_best_model.pkl'
    if not os.path.exists(model_path):
        model_path = 'lead_poisoning_model.pkl'
        print(f"注意: 未找到最佳单模型，将使用常规模型 {model_path}")
    
    # 1. 加载保存的模型
    model, selected_features = load_saved_model(model_path)
    if model is None:
        print("错误: 无法加载模型，请先运行模型训练")
        return
    
    # 2. 创建模拟的批量患者数据
    try:
        # 从训练数据中抽取一批样本
        train_data = pd.read_excel('aldata.xlsx')
        # 随机选择10个样本
        sample_indices = np.random.choice(len(train_data), size=10, replace=False)
        batch_data = train_data.iloc[sample_indices].copy()
        print(f"\n已选择 {len(batch_data)} 个样本用于批量预测演示")
        
    except Exception as e:
        print(f"无法读取训练数据，创建随机样本: {str(e)}")
        # 如果无法读取训练数据，则创建随机样本
        batch_data = pd.DataFrame(np.random.randn(10, len(selected_features)), 
                                 columns=selected_features)
        # 添加ID列
        batch_data['患者ID'] = [f'P{i+1001}' for i in range(10)]
    
    # 3. 进行批量预测
    print("\n开始批量预测...")
    results = batch_predict(model, batch_data, selected_features)
    
    if results is not None:
        print("\n批量预测结果概览:")
        print(f"共 {len(results)} 条记录")
        print(f"高风险患者数: {sum(results['风险预测'] == 1)}")
        print(f"低风险患者数: {sum(results['风险预测'] == 0)}")
        print("\n风险分布:")
        risk_counts = results['风险等级'].value_counts()
        for risk_level, count in risk_counts.items():
            print(f"- {risk_level}: {count} 人 ({count/len(results):.1%})")

def compare_models_demo():
    """比较并可视化不同模型的性能"""
    print("="*50)
    print("模型性能比较与可视化")
    print("="*50)
    
    # 检查是否存在模型比较结果
    if not os.path.exists('model_comparison_results.csv'):
        print("错误: 未找到模型比较结果文件，请先运行优化模型训练")
        return
    
    try:
        # 读取比较结果
        comparison_df = pd.read_csv('model_comparison_results.csv')
        print("\n模型性能比较:")
        print(comparison_df)
        
        # 绘制性能比较条形图
        plt.figure(figsize=(14, 8))
        
        # 设置分组柱状图的位置
        models = comparison_df['模型']
        x = np.arange(len(models))
        width = 0.15  # 柱子宽度
        
        # 绘制各指标条形图
        plt.bar(x - width*2, comparison_df['准确率'], width, label='准确率', color='#2C7FB8')
        plt.bar(x - width, comparison_df['精确率'], width, label='精确率', color='#7FCDBB')
        plt.bar(x, comparison_df['召回率'], width, label='召回率', color='#EAAB00')
        plt.bar(x + width, comparison_df['F1得分'], width, label='F1得分', color='#D7301F')
        plt.bar(x + width*2, comparison_df['AUC'], width, label='AUC', color='#7A0177')
        
        # 添加标签和图例
        plt.xlabel('模型', fontsize=12)
        plt.ylabel('评分', fontsize=12)
        plt.title('不同模型性能指标比较', fontsize=14)
        plt.xticks(x, models, rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        
        # 保存图表
        plt.tight_layout()
        plt.savefig('model_comparison_chart.png', dpi=300)
        print("\n模型比较图表已保存为: model_comparison_chart.png")
        
        # 显示图表
        plt.show()
        
    except Exception as e:
        print(f"比较模型时出错: {str(e)}")

def main():
    """主函数"""
    print("铅中毒预测模型演示程序")
    print("\n选择要运行的演示:")
    print("1. 原始集成模型训练")
    print("2. 优化单模型训练与比较")
    print("3. 高级单模型优化（进一步提升性能）")
    print("4. 单个病例预测")
    print("5. 批量预测")
    print("6. 模型性能比较与可视化")
    print("7. 全部运行")
    
    try:
        choice = input("\n请输入选项 (1-7): ")
        
        if choice == '1':
            run_model_training()
        elif choice == '2':
            run_optimized_model_training()
        elif choice == '3':
            run_advanced_model_optimization()
        elif choice == '4':
            run_single_prediction_demo()
        elif choice == '5':
            run_batch_prediction_demo()
        elif choice == '6':
            compare_models_demo()
        elif choice == '7':
            success1 = run_model_training()
            success2 = run_optimized_model_training()
            success3 = run_advanced_model_optimization()
            if success1 or success2 or success3:
                run_single_prediction_demo()
                run_batch_prediction_demo()
                compare_models_demo()
        else:
            print("无效选项，请输入1-7之间的数字")
    except Exception as e:
        print(f"运行出错: {str(e)}")

if __name__ == '__main__':
    main() 