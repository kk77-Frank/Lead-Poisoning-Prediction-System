# 🏥 Lead Poisoning Risk Assessment System

An intelligent risk assessment system for lead poisoning using machine learning, available in both Flask and Streamlit versions.

## 🌟 Features

- **AI-Powered Prediction**: Advanced machine learning model for accurate risk assessment
- **Single Patient Analysis**: Manual input for individual patient risk evaluation
- **Batch Processing**: Upload Excel/CSV files for multiple patient analysis
- **Interactive Visualizations**: Real-time charts and graphs
- **Clinical Recommendations**: AI-generated personalized treatment suggestions
- **Export Functionality**: Download results in Excel format
- **Responsive Design**: Works on desktop and mobile devices

## 🚀 Quick Start

### Streamlit Version (Recommended)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run streamlit_app.py

# Or use the automated deployment script
python deploy.py
```

Access the application at `http://localhost:8501`

### Flask Version

```bash
# Install dependencies
pip install flask flask-cors pandas openpyxl

# Run the application
python app.py
```

Access the application at `http://localhost:5000`

## 📊 System Overview

### Risk Levels
- **Low Risk (0-40%)**: Routine health management
- **Medium Risk (40-70%)**: Regular monitoring required  
- **High Risk (70-80%)**: Close monitoring and preventive treatment
- **Extremely High Risk (80%+)**: Immediate intervention required

### Model Requirements
- **Input Features**: 39 clinical and demographic features
- **Key Features**: Age, gender, blood lead levels, hospital days, admission history
- **Output**: Risk probability and clinical recommendations

## 🛠️ Deployment Options

### 1. Local Development
- Direct Python execution
- Suitable for development and testing

### 2. Streamlit Cloud
- Free hosting on Streamlit Cloud
- Easy GitHub integration
- Automatic deployments

### 3. Docker
- Containerized deployment
- Consistent environment
- Easy scaling

### 4. Heroku
- Cloud platform deployment
- Automatic scaling
- Custom domain support

## 📁 Project Structure

```
├── streamlit_app.py              # Streamlit application
├── app.py                        # Flask application
├── requirements.txt              # Dependencies
├── deploy.py                     # Deployment script
├── Dockerfile                    # Docker configuration
├── docker-compose.yml            # Docker Compose
├── templates/                    # HTML templates (Flask)
├── static/                       # CSS/JS files (Flask)
├── .streamlit/                   # Streamlit configuration
├── lead_poisoning_optimized_model.pkl  # ML model
└── README.md                     # This file
```

## 🔧 Configuration

### Environment Variables
- `STREAMLIT_SERVER_PORT`: Server port (default: 8501)
- `FLASK_PORT`: Flask server port (default: 5000)

### Model File
Ensure `lead_poisoning_optimized_model.pkl` is in the project root directory.

## 📈 Usage Examples

### Single Patient Prediction
1. Navigate to "Single Patient Prediction"
2. Fill in patient information
3. Click "Predict Risk"
4. View results and recommendations

### Batch Processing
1. Navigate to "Batch Prediction"
2. Download the data template
3. Fill in patient data
4. Upload the file
5. View results and download report

## 🛡️ Security & Privacy

- Patient data is processed locally
- No data is stored permanently
- HTTPS recommended for production
- Compliance with healthcare data regulations

## ⚠️ Disclaimer

This system is designed to assist healthcare professionals in clinical decision-making. It should not replace professional medical judgment or be used as the sole basis for treatment decisions.

## 📄 License

This project is licensed under the MIT License.
