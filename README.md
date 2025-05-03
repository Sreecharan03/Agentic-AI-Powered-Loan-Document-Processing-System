
# 🧾 Agentic AI-Powered Loan Document Processing System  
**End-to-End OCR + LLM + Schema Validation for Financial Document Automation**

This repository provides a production-ready pipeline for automated personal loan document processing using an **Agentic AI architecture**. It combines **OCR (PaddleOCR)**, **Vision LLMs (LLaVA)**, **schema-based validation**, and **cross-document consistency checks** tailored for Indian financial documents such as Aadhaar, PAN, salary slips, bank statements, and property records.

---

## 🚀 Features

- 🔍 **OCR Field Detection** using PaddleOCR (multilingual, structured)
- 🧠 **Vision-Language Understanding** via LLaVA (for overlays, annotations)
- ✅ **Schema-Driven Validation** using `schema.json`
- 🔄 **Cross-Document Inference Agent** to catch mismatches & fraud
- 🧾 **Summary Agent** for document status & insights
- 📊 **Explainable Output** with structured JSON, flags & scores
- 🔐 **Compliance Ready** for TCS/NBFC standards (e.g., KYC norms)

---

## 📁 Project Structure

```
agentic-ocr-loan-processor/
├── agents/
│   ├── extractor_agent.py
│   ├── validator_agent.py
│   ├── crosscheck_agent.py
│   ├── summarizer_agent.py
│   └── compliance_agent.py
│
├── schemas/
│   ├── schema_identity.json
│   ├── schema_income.json
│   ├── schema_bank.json
│   └── schema_property.json
│
├── samples/
│   ├── aadhaar_sample.png
│   ├── pan_card_sample.png
│   └── bank_statement_sample.pdf
│
├── utils/
│   ├── paddle_ocr_wrapper.py
│   ├── llava_vision_parser.py
│   └── json_validator.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

## 📜 Sample Schema: `schema_identity.json`

```json
{
  "document_type": "identity",
  "required_fields": [
    {"field": "name", "type": "string"},
    {"field": "dob", "type": "date"},
    {"field": "gender", "type": "string"},
    {"field": "aadhaar_number", "type": "string", "regex": "^[2-9]{1}[0-9]{11}$"},
    {"field": "address", "type": "string"}
  ],
  "field_validations": {
    "aadhaar_number": {
      "length": 12,
      "numeric_only": true
    },
    "dob": {
      "format": "DD-MM-YYYY"
    }
  }
}
```

---

## 🧠 How It Works (Agentic Flow)

```
OCR Image → [Extractor Agent] → Raw JSON → [Validator Agent] → Cleaned JSON
                 ↓
            [CrossCheck Agent] ← Multiple Docs
                 ↓
            [Summarizer Agent] → Final JSON Summary
                 ↓
           [Compliance Agent] → Pass/Fail with Reasons
```

---

## 🛠️ Setup Instructions

```bash
# Create environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the main agentic pipeline
python main.py --input_dir samples/ --output_dir results/
```

---

## ✅ Example Output

```json
{
  "document": "aadhaar_sample.png",
  "status": "valid",
  "extracted_fields": {
    "name": "RAHUL SHARMA",
    "dob": "21-01-1995",
    "gender": "Male",
    "aadhaar_number": "289485902192",
    "address": "Mumbai, Maharashtra"
  },
  "flags": [],
  "summary": "Identity document is complete and matches all required fields."
}
```

---

## 📈 Future Enhancements

- Integration with LLM chatbots for live document Q&A
- Face matching with Aadhaar photo
- Fraud score prediction via ML
- API endpoint for NBFC onboarding

---

## 👨‍⚖️ License

MIT License – for research and non-commercial use only. Commercial users must request explicit permission.
