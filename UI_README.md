# 🎨 Streamlit Web UI Guide

A beautiful, interactive web interface for the Explainable ML Agentic Pipeline!

---

## 🚀 Quick Start

### **1. Install Streamlit**

```powershell
pip install streamlit
```

(Already included in `requirements.txt` if you ran `pip install -r requirements.txt`)

### **2. Run the Web UI**

```powershell
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

---

## 📱 Features

### **📤 Tab 1: Upload & Run**

- **Upload CSV files** via drag & drop or file browser
- **Preview dataset** with automatic statistics
- **Select target column** interactively
- **Configure settings** in the sidebar:
  - Task type (classification/regression)
  - Domain (healthcare/finance/general)
  - Enable/disable LLM reasoning
  - Select ML algorithms
  - Adjust hyperparameters
- **Run pipeline** with a single click
- **Real-time progress** updates

### **📊 Tab 2: Results**

- **Status cards**: Model approved? Deployment ready?
- **Performance metrics**: Interactive charts
- **Confusion matrix**: For classification tasks
- **Detailed metrics**: Expandable JSON view
- **Recommendations**: AI-generated suggestions
- **Download results**: JSON format
- **Save model**: One-click model export

### **🤖 Tab 3: Agent Communication**

- **Message timeline**: See all agent interactions
- **Message types**: Color-coded (request, response, decision, error)
- **Full audit trail**: Complete communication history

### **📈 Tab 4: Visualizations**

- **Performance trends**: Across iterations
- **Execution time**: Breakdown by stage
- **Interactive plots**: Powered by Plotly

---

## 🎨 Screenshots

### Home Screen

![Home](https://img.shields.io/badge/Upload-Dataset-blue?style=for-the-badge)

### Results Dashboard

![Results](https://img.shields.io/badge/View-Results-green?style=for-the-badge)

### Agent Communication

![Agents](https://img.shields.io/badge/Agent-Messages-orange?style=for-the-badge)

---

## ⚙️ Configuration Options

### **Sidebar Settings:**

#### **🧠 LLM Settings**

- **Enable LLM Reasoning**: Toggle LLM-powered insights
- **Model Selection**: Choose from:
  - `llama3.1:8b` (recommended)
  - `llama3.1:70b` (better quality, slower)
  - `mistral:7b` (faster)

#### **🤖 Agent Settings**

- **Algorithms**: Multi-select from:
  - Random Forest
  - XGBoost
  - Logistic Regression
  - SVM
- **CV Folds**: 3-10 (default: 5)
- **Performance Threshold**: 0.5-1.0 (default: 0.75)
- **Max Retrain Cycles**: 1-5 (default: 3)

---

## 📝 Usage Examples

### **Example 1: Heart Disease Prediction**

1. **Upload**: `data/heart.csv`
2. **Select Target**: `target`
3. **Set Domain**: `Healthcare`
4. **Enable LLM**: ✅
5. **Click**: `🚀 Run ML Pipeline`
6. **Wait**: ~5-15 minutes (with LLM)
7. **View Results**: Check metrics, confusion matrix
8. **Download**: Save results and model

### **Example 2: Quick Test (No LLM)**

1. **Upload**: Any CSV file
2. **Select Target**: Your target column
3. **Disable LLM**: ❌ (faster)
4. **Click**: `🚀 Run ML Pipeline`
5. **Wait**: ~30-60 seconds
6. **View Results**: Instant feedback

---

## 🎯 Tips & Best Practices

### **Performance**

1. **Disable LLM** for quick testing (10x faster)
2. **Reduce algorithms** to 1-2 for speed
3. **Lower CV folds** (3 instead of 5) for faster runs
4. **Use smaller datasets** (<10K rows) for testing

### **Best Results**

1. **Enable LLM** for explainable insights
2. **Clean your data** before upload
3. **Select appropriate domain** (healthcare/finance)
4. **Check target distribution** before running
5. **Review recommendations** after completion

### **Troubleshooting**

**Issue**: "LLM error"

- **Fix**: Make sure Ollama is running (`ollama serve`)

**Issue**: "Model not approved"

- **Fix**: Check data quality, try different algorithms

**Issue**: "Slow execution"

- **Fix**: Disable LLM or reduce dataset size

---

## 🔧 Advanced Configuration

### **Custom Themes**

Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#1E88E5"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### **Port Configuration**

Run on different port:

```powershell
streamlit run app.py --server.port 8502
```

### **Debug Mode**

```powershell
streamlit run app.py --logger.level debug
```

---

## 📊 Supported Datasets

### **Format Requirements**

- ✅ CSV files
- ✅ Headers required
- ✅ Mixed data types (numeric + categorical)
- ✅ Missing values (automatically handled)

### **Size Limits**

- **Recommended**: < 10K rows
- **Maximum**: < 100K rows
- **Columns**: < 100

### **Target Column**

- **Classification**: 2+ unique values
- **Regression**: Continuous numeric values

---

## 🎨 UI Components

### **Metrics Cards**

Beautiful gradient cards showing key metrics

### **Interactive Charts**

- Bar charts for metrics
- Heatmaps for confusion matrices
- Line charts for trends
- Pie charts for time distribution

### **Status Boxes**

- ✅ **Success** (green): Model approved
- ⚠️ **Warning** (yellow): Action needed
- ❌ **Error** (red): Issues detected

### **Agent Cards**

Color-coded message timeline:

- 🔵 **Request**: Agent asks for input
- 🟢 **Response**: Agent provides output
- 🟠 **Decision**: Judge agent ruling
- 🔴 **Error**: Something went wrong

---

## 🚀 Deployment

### **Local Network Access**

```powershell
streamlit run app.py --server.address 0.0.0.0
```

Access from other devices: `http://YOUR_IP:8501`

### **Cloud Deployment**

**Streamlit Cloud** (Free):

1. Push code to GitHub
2. Go to https://streamlit.io/cloud
3. Connect repository
4. Deploy!

**Docker**:

```dockerfile
FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

---

## 💡 Keyboard Shortcuts

- `Ctrl + R`: Rerun app
- `Ctrl + Shift + R`: Clear cache and rerun
- `Ctrl + K`: Open command palette
- `Ctrl + /`: Toggle sidebar

---

## 📚 Resources

- **Streamlit Docs**: https://docs.streamlit.io/
- **Plotly Docs**: https://plotly.com/python/
- **Pipeline Docs**: See `README.md`

---

## ❓ FAQ

**Q: How long does a pipeline run take?**  
A: 30-60 seconds without LLM, 5-15 minutes with LLM

**Q: Can I use my own data?**  
A: Yes! Upload any CSV file

**Q: Is my data stored?**  
A: No, data is only temporarily stored during processing

**Q: Can I run multiple pipelines?**  
A: Yes, but run them sequentially (one at a time)

**Q: Can I compare results?**  
A: Download results JSON after each run and compare manually

---

## 🎉 Getting Started NOW

```powershell
# 1. Make sure dependencies are installed
pip install streamlit

# 2. Start the UI
streamlit run app.py

# 3. Open browser (auto-opens)
# http://localhost:8501

# 4. Upload your CSV
# 5. Select target column
# 6. Click "Run ML Pipeline"
# 7. View results!
```

---

**Enjoy your beautiful ML pipeline interface!** 🚀

**Need help?** Check the main `README.md` or open an issue.
