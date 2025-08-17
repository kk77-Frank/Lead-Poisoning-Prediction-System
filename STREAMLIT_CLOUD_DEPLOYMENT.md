# 🚀 Streamlit Cloud 部署指南

## 📋 必需上传到GitHub的文件清单

### ✅ 核心文件（必须上传）

1. **`streamlit_app.py`** - 主应用文件
2. **`requirements.txt`** - 依赖包列表（已更新包含所有ML库）
3. **`lead_poisoning_optimized_model.pkl`** - 机器学习模型文件

### 🔧 配置文件（推荐上传）

4. **`.streamlit/config.toml`** - Streamlit配置
5. **`README_EN.md`** - 项目说明文档

### 📚 文档文件（可选）

6. **`STREAMLIT_DEPLOYMENT.md`** - 完整部署指南
7. **`STREAMLIT_CLOUD_DEPLOYMENT.md`** - 本文件

## 🚨 重要检查事项

### 1. 模型文件大小
- Streamlit Cloud 限制单个文件最大 100MB
- 如果模型文件过大，需要使用 Git LFS 或外部存储

### 2. 依赖包版本
更新后的 `requirements.txt` 包含：
```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
joblib>=1.1.0
plotly>=5.0.0
openpyxl>=3.0.0
xlrd>=2.0.0
xgboost>=1.6.0
lightgbm>=3.3.0
catboost>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

### 3. 文件结构检查
确保您的GitHub仓库包含以下结构：
```
your-repo/
├── streamlit_app.py
├── requirements.txt
├── lead_poisoning_optimized_model.pkl
├── .streamlit/
│   └── config.toml
├── README_EN.md
└── STREAMLIT_CLOUD_DEPLOYMENT.md
```

## 📤 部署步骤

### 第1步：准备GitHub仓库

1. **创建新的GitHub仓库**
   - 登录 GitHub
   - 点击 "New repository"
   - 命名为 `lead-poisoning-risk-assessment`
   - 设置为 Public（Streamlit Cloud 免费版需要公开仓库）

2. **上传文件到仓库**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Lead Poisoning Risk Assessment System"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/lead-poisoning-risk-assessment.git
   git push -u origin main
   ```

### 第2步：部署到Streamlit Cloud

1. **访问 Streamlit Cloud**
   - 打开 [share.streamlit.io](https://share.streamlit.io)
   - 使用GitHub账号登录

2. **创建新应用**
   - 点击 "New app"
   - 选择您的GitHub仓库
   - 主文件路径：`streamlit_app.py`
   - 点击 "Deploy!"

3. **等待部署完成**
   - 初次部署可能需要5-10分钟
   - 系统会自动安装依赖包
   - 部署成功后会获得一个公开URL

## 🔍 故障排除

### 常见问题及解决方案

1. **模块导入错误**
   - 检查 `requirements.txt` 是否包含所有依赖
   - 确保版本号兼容

2. **模型文件加载失败**
   - 确认模型文件已上传到仓库
   - 检查文件路径是否正确
   - 验证文件大小是否超过限制

3. **内存不足错误**
   - 优化模型文件大小
   - 使用模型压缩技术
   - 考虑升级到付费版本

4. **部署超时**
   - 检查依赖包是否过多
   - 简化 requirements.txt
   - 重新触发部署

## 🎯 部署后验证

部署成功后，请验证以下功能：

- [ ] 应用正常加载
- [ ] 模型成功加载
- [ ] 单患者预测功能正常
- [ ] 文件上传功能正常
- [ ] 批量预测功能正常
- [ ] 图表显示正常
- [ ] 结果导出功能正常

## 🔗 有用链接

- [Streamlit Cloud 文档](https://docs.streamlit.io/streamlit-cloud)
- [Streamlit 社区论坛](https://discuss.streamlit.io/)
- [GitHub 帮助文档](https://docs.github.com/)

## 📞 技术支持

如果遇到部署问题：
1. 检查 Streamlit Cloud 的日志输出
2. 参考官方文档
3. 在社区论坛寻求帮助
4. 联系开发团队

---

**注意**: 免费版 Streamlit Cloud 有一些限制，包括计算资源和并发用户数。对于生产环境，建议考虑付费版本或其他部署选项。
