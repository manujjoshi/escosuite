# 🔧 Troubleshooting: N/A Results Issue

## Problem: RFP Analyzer Returns N/A Results

If you're getting N/A results, it's likely because:
1. **Company knowledge bases are missing**
2. **Ollama is not running**
3. **PDF extraction failed**

## ✅ Quick Fix (3 Steps)

### Step 1: Create Company Knowledge Bases

Run the setup script:
```bash
python setup_company_kb.py
```

This will create `company_db/IKIO.json`, `company_db/METCO.json`, and `company_db/SUNSPRINT.json` with sample data.

**Or manually create them:**

Create `company_db/IKIO.json`:
```json
{
  "texts": [
    "IKIO Energy specializes in solar energy projects.",
    "We are licensed in all 50 states.",
    "We have completed over 500 MW of installations.",
    "We maintain $10 million general liability insurance.",
    "Our bonding capacity is up to $50 million per project.",
    "We have experience with BABA compliance.",
    "We are certified as MBE and SBE.",
    "Our team includes 50+ engineers and 200+ technicians.",
    "We have partnerships with Tier 1 solar manufacturers.",
    "We maintain ISO 9001 and ISO 14001 certifications.",
    "OSHA 30-hour certified supervisors on all projects.",
    "25-year performance warranty on all installations.",
    "We provide 24/7 O&M services.",
    "Strong financial backing from institutional investors.",
    "Experience with Davis-Bacon prevailing wages.",
    "We follow strict environmental protocols.",
    "Project timelines typically 6-12 months.",
    "Completed federal projects for DOD and DOE.",
    "Perfect safety record for past 5 years.",
    "We use Primavera P6 and Procore for project management."
  ]
}
```

Create similar files for METCO and SUNSPRINT.

### Step 2: Verify Ollama is Running

```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# If not running, start it
ollama serve

# Pull the model if needed
ollama pull llama3

# Verify model is downloaded
ollama list
```

### Step 3: Test the Upload

1. Go to `http://localhost:5001/rfp-analyzer/`
2. Upload an RFP PDF
3. **Important**: Also upload company context files for IKIO, METCO, and SUNSPRINT (TXT or DOCX files)
   - Or ensure the JSON files exist in `company_db/` directory

## 🔍 Detailed Diagnostics

### Check 1: Company Knowledge Bases Exist

```bash
# Check if files exist
ls -la company_db/

# Should show:
# IKIO.json
# METCO.json
# SUNSPRINT.json
```

**Windows:**
```cmd
dir company_db
```

### Check 2: Ollama Service Status

```bash
# Test Ollama
curl http://localhost:11434/api/version

# Expected response:
# {"version":"0.x.x"}
```

**If Ollama is not responding:**
```bash
# Start Ollama service
ollama serve
```

### Check 3: Check Flask Logs

When you run the analyzer, check your Flask console for errors:
- "Company knowledge base missing" - Run `setup_company_kb.py`
- "Ollama error" - Check Ollama is running
- "Could not extract text from PDF" - PDF might be corrupted or image-only

### Check 4: Test with Sample Companies

Upload company context directly through the web interface:

1. Create a simple text file `ikio_context.txt`:
```
IKIO Energy is a solar company with 10+ years experience.
We are licensed in all states.
We have $10M insurance coverage.
We maintain ISO certifications.
We have completed 500+ MW of solar projects.
```

2. Upload this file in the IKIO Context field
3. Do the same for METCO and SUNSPRINT

## 🎯 Common Error Messages & Solutions

### Error: "No company knowledge bases found"

**Solution:**
```bash
python setup_company_kb.py
```

Or upload context files through the web interface.

### Error: "Failed to summarize RFP. Please check Ollama is running."

**Solution:**
```bash
ollama serve
ollama pull llama3
```

### Error: "Could not extract text from PDF"

**Solution:**
- Ensure PDF is not password-protected
- Try a different PDF
- Check if PDF contains actual text (not just images)

### Error: "Ollama connection refused"

**Solution:**
```bash
# Check Ollama status
curl http://localhost:11434/api/version

# If not running, start it
ollama serve

# In another terminal, pull model
ollama pull llama3
```

## 📝 Verification Checklist

Before running the analyzer, verify:

- [ ] `company_db/` directory exists
- [ ] `company_db/IKIO.json` exists and has content
- [ ] `company_db/METCO.json` exists and has content
- [ ] `company_db/SUNSPRINT.json` exists and has content
- [ ] Ollama is running (`curl http://localhost:11434/api/version`)
- [ ] Llama3 model is downloaded (`ollama list`)
- [ ] Flask app is running without errors
- [ ] You're logged in as admin

## 🧪 Test with Debug Mode

Add this to check what's happening:

```python
# In rfp_analyzer_routes.py, add logging:
import logging
logging.basicConfig(level=logging.DEBUG)
```

Then check Flask console output when analyzing.

## 💡 Quick Test Command

```bash
# Test Ollama directly
curl -X POST http://localhost:11434/api/generate -d '{
  "model": "llama3",
  "prompt": "Say hello",
  "stream": false
}'
```

Expected response should include "hello" in the response.

## 🔧 Advanced Troubleshooting

### Reset Everything

```bash
# 1. Stop Flask
Ctrl+C

# 2. Stop Ollama
killall ollama  # Linux/Mac
# Or close Ollama app on Windows

# 3. Clear company_db
rm -rf company_db/*

# 4. Recreate knowledge bases
python setup_company_kb.py

# 5. Start Ollama
ollama serve

# 6. Verify model
ollama pull llama3
ollama list

# 7. Restart Flask
python app_v2.py
```

### Check File Permissions

```bash
# Ensure company_db is writable
chmod -R 755 company_db/

# Ensure JSON files are readable
chmod 644 company_db/*.json
```

## 📊 Expected Flow

```
1. User uploads RFP PDF
   ↓
2. Extract text (PyMuPDF or OCR)
   ✅ Should get multiple pages of text
   ↓
3. Summarize with Ollama
   ✅ Should get project_type, scope, etc.
   ↓
4. Load company knowledge bases
   ✅ Should load 3 JSON files
   ↓
5. Score each company
   ✅ Should evaluate 23 checklist items per company
   ↓
6. Display results
   ✅ Should show scores, compliance %, decision
```

## 🎯 Still Not Working?

### Get Detailed Error Info

The updated code now returns detailed error messages. Check the browser console (F12) for:
- `error` field in JSON response
- `traceback` field with Python stack trace
- `missing_companies` field showing which companies failed

### Manual Test

```python
# Test in Python console
import json
import os

# Check if files exist
for company in ["IKIO", "METCO", "SUNSPRINT"]:
    path = f"company_db/{company}.json"
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
            print(f"{company}: {len(data.get('texts', []))} texts")
    else:
        print(f"{company}: MISSING")
```

### Contact Info

If still having issues, provide:
1. Flask console output
2. Browser console errors (F12)
3. Output of `ollama list`
4. Output of `ls company_db/`
5. Python version (`python --version`)

---

## ✅ Success Criteria

You'll know it's working when:
- ✅ Upload completes without errors
- ✅ Loading spinner shows for 2-5 minutes
- ✅ Results section displays with actual data (not N/A)
- ✅ Winner card shows company name and compliance %
- ✅ Individual company cards show scores
- ✅ Detailed table shows Yes/No/Partial evaluations

**Key**: The most common issue is **missing company knowledge bases**. Run `python setup_company_kb.py` first!

