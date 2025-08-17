import joblib

# 加载模型文件
try:
    model_data = joblib.load('d:/Frank/new/lead_poisoning_optimized_model.pkl')
    
    # 提取特征列表
    if isinstance(model_data, dict) and 'selected_features' in model_data:
        selected_features = model_data['selected_features']
        print("模型使用的特征列表:")
        for i, feature in enumerate(selected_features, 1):
            print(f"{i}. {feature}")
        print(f"\n总计: {len(selected_features)}个特征")
    else:
        print("模型文件中没有找到特征列表信息")
        
except Exception as e:
    print(f"加载模型文件失败: {str(e)}")