# Lead Poisoning Risk Assessment System - Streamlit Deployment

## 🚀 Quick Start

### Option 1: Using the Deploy Script (Recommended)

1. **Run the deployment script**
   ```bash
   python deploy.py
   ```
   This will automatically:
   - Check for required files
   - Install dependencies
   - Create configuration
   - Start the application

### Option 2: Manual Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Access the Application**
   Open your browser and go to `http://localhost:8501`

## 📁 File Structure

```
├── streamlit_app.py              # Main Streamlit application
├── requirements.txt              # Python dependencies
├── deploy.py                     # Automated deployment script
├── Dockerfile                    # Docker configuration
├── docker-compose.yml            # Docker Compose configuration
├── .streamlit/
│   └── config.toml              # Streamlit configuration
├── lead_poisoning_optimized_model.pkl  # ML model file
├── app.py                        # Original Flask application (for reference)
├── templates/                    # Original HTML templates (for reference)
├── static/                       # Original CSS/JS files (for reference)
└── STREAMLIT_DEPLOYMENT.md      # This file
```

## 🌐 Deployment Options

### 1. Streamlit Cloud (Recommended)

1. **Push to GitHub**
   - Create a new repository on GitHub
   - Push all files to the repository

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository
   - Choose `streamlit_app.py` as the main file
   - Click "Deploy"

### 2. Heroku Deployment

1. **Create Heroku App**
   ```bash
   heroku create your-app-name
   ```

2. **Add Buildpack**
   ```bash
   heroku buildpacks:set heroku/python
   ```

3. **Create Procfile**
   ```
   web: sh setup.sh && streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
   ```

4. **Create setup.sh**
   ```bash
   mkdir -p ~/.streamlit/
   echo "\
   [general]\n\
   email = \"your-email@domain.com\"\n\
   " > ~/.streamlit/credentials.toml
   echo "\
   [server]\n\
   headless = true\n\
   enableCORS=false\n\
   port = $PORT\n\
   " > ~/.streamlit/config.toml
   ```

5. **Deploy**
   ```bash
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

### 3. Docker Deployment

#### Option A: Using Docker Compose (Recommended)

1. **Run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

2. **Run in background**
   ```bash
   docker-compose up -d --build
   ```

3. **Stop the application**
   ```bash
   docker-compose down
   ```

#### Option B: Using Docker directly

1. **Build the image**
   ```bash
   docker build -t lead-poisoning-app .
   ```

2. **Run the container**
   ```bash
   docker run -p 8501:8501 lead-poisoning-app
   ```

## 🔧 Configuration

### Environment Variables

- `STREAMLIT_SERVER_PORT`: Server port (default: 8501)
- `STREAMLIT_SERVER_ADDRESS`: Server address (default: localhost)

### Model File

Ensure the model file `lead_poisoning_optimized_model.pkl` is in the same directory as `streamlit_app.py`.

## 📊 Features

- **Single Patient Prediction**: Manual input for individual risk assessment
- **Batch Prediction**: Upload Excel/CSV files for multiple patients
- **Data Visualization**: Interactive charts and graphs
- **Export Results**: Download prediction results
- **Responsive Design**: Works on desktop and mobile devices

## 🛠️ Troubleshooting

### Common Issues

1. **Model Loading Error**
   - Ensure `lead_poisoning_optimized_model.pkl` exists
   - Check file permissions
   - Verify model compatibility

2. **File Upload Issues**
   - Check file format (Excel/CSV)
   - Verify file size limits
   - Ensure proper column names

3. **Memory Issues**
   - Reduce batch size for large files
   - Consider upgrading deployment resources

### Performance Optimization

- Use `@st.cache_resource` for model loading
- Implement data caching for large datasets
- Optimize chart rendering for better performance

## 📞 Support

For technical support or questions:
- Check the Streamlit documentation: [docs.streamlit.io](https://docs.streamlit.io)
- Review error logs in the deployment platform
- Contact the development team

## 🔒 Security Considerations

- Ensure patient data privacy compliance
- Use HTTPS in production
- Implement proper authentication if needed
- Regular security updates for dependencies
