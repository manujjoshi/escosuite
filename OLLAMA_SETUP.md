# Ollama Setup Instructions for Bid Analyzer

This Bid Analyzer now uses **Ollama** instead of OpenAI API, eliminating the need for API keys and providing local, free AI inference.

## 🚀 Quick Setup

### 1. Install Ollama

#### Windows
- Download from: https://ollama.ai/download
- Run the installer
- Ollama will start automatically on `http://localhost:11434`

#### Linux/Mac
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Pull the Recommended Model

Open your terminal/command prompt and run:

```bash
ollama pull llama3
```

**Alternative models you can use:**
- `ollama pull llama3.1` - Larger, more accurate
- `ollama pull mistral` - Faster, good balance
- `ollama pull llama2` - Lighter weight
- `ollama pull mixtral` - High performance

### 3. Verify Ollama is Running

```bash
ollama list
```

You should see the models you've downloaded.

### 4. Configure Environment Variables (Optional)

Create a `.env` file in the project directory:

```env
# Default is http://localhost:11434
OLLAMA_URL=http://localhost:11434

# Default is llama3
OLLAMA_MODEL=llama3
```

### 5. Install Python Dependencies

```bash
pip install requests
```

## 🎯 Usage

Run the Bid Analyzer as usual:

```bash
streamlit run bid_analyser_V6.py
```

The app will now use your local Ollama instance instead of OpenAI!

## 🔧 Troubleshooting

### "Connection refused" error
- Make sure Ollama is running: `ollama serve`
- Check if port 11434 is accessible

### Slow responses
- Try a lighter model: `ollama pull mistral`
- Increase timeout in code if needed

### JSON parsing errors
- Some models work better with JSON output
- Recommended: `llama3`, `llama3.1`, `mistral`

## 📊 Model Comparison

| Model | Size | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| llama3 | 4.7GB | Medium | High | **Recommended for bid analysis** |
| llama3.1 | 8.5GB | Slow | Very High | Complex RFPs |
| mistral | 4.1GB | Fast | Good | Quick evaluations |
| llama2 | 3.8GB | Very Fast | Medium | Testing |

## 💡 Benefits of Using Ollama

✅ **No API Costs** - Completely free, runs locally  
✅ **Privacy** - Your RFP data never leaves your machine  
✅ **No Rate Limits** - Process unlimited RFPs  
✅ **Offline Capable** - Works without internet after model download  
✅ **Fast** - No network latency, runs on your hardware  

## 🆘 Support

For Ollama issues: https://github.com/ollama/ollama/issues  
For Bid Analyzer issues: Contact your system administrator

