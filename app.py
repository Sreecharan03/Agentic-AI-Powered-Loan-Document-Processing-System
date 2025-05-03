# --- IMPORTS ---
import streamlit as st
from PIL import Image
import json
import io
import os
import re
import time
import textwrap
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

# --- PaddleOCR Import ---
# Suppress PaddleOCR logging unless it's an error
os.environ['PP_DISABLE_LOGGING'] = '1' # Doesn't always work, but worth trying
try:
    from paddleocr import PaddleOCR
    paddleocr_imported = True
    # Don't print success here, do it in init function
except ImportError as e:
    # Error will be handled during initialization
    paddleocr_imported = False
    # Define a dummy class if import fails
    class PaddleOCR:
        def __init__(self, **kwargs): pass
        def ocr(self, img_path, cls=True):
            # Return format similar to paddleocr on empty/error
            return [[],]

# --- Other Imports ---
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig # Added BitsAndBytes

# --- Schema Validator Import ---
# Attempt import, handle failure during validation step
try:
    from src.schema_validator import validate_extraction
    validator_imported = True
except ImportError as e:
    validator_imported = False
    # Define a dummy validator function if import fails
    def validate_extraction(data: Any, document_type: str) -> Tuple[bool, str]:
        return False, f"Validation skipped: Validator module not found or failed to import ({e}). Ensure src/schema_validator.py exists or adjust import."
except Exception as e:
    # Catch other unexpected errors during import
    validator_imported = False
    def validate_extraction(data: Any, document_type: str) -> Tuple[bool, str]:
        return False, f"Validation skipped: Error during validator import ({e})."


# --- Configuration ---
# Determine BASE_DIR relative to this script file
BASE_DIR = Path(__file__).resolve().parent
SCHEMAS_DIR = BASE_DIR / "schemas"

# Define SCHEMA_MAP using the resolved BASE_DIR
# Ensure your schema files are in a 'schemas' subdirectory relative to app.py
SCHEMA_MAP = {
    "loan_application": SCHEMAS_DIR / "loan_application.json",
    "property": SCHEMAS_DIR / "property.json",
    "bank_statements": SCHEMAS_DIR / "bank_statements.json",
    "income_salary_slip": SCHEMAS_DIR / "income_salary_slip.json",
    # Add other document types and their corresponding schema paths here
}
DEFAULT_DOC_TYPE = "bank_statements" # Or choose another default

# LLM Configuration
MAX_TOKENS = 2048 # Increased max tokens for potentially complex JSON
HF_TOKEN = None # Set your Hugging Face token string here if needed (e.g., from st.secrets)
MODEL_ID = "HuggingFaceH4/zephyr-7b-beta" # LLM for structuring the OCR text
# MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.1" # Alternative

# --- HELPER FUNCTIONS (Copied from your script, minor adjustments) ---

def load_schema(schema_path: Path) -> Optional[Dict[str, Any]]:
    """Loads a JSON schema file."""
    if not schema_path.is_file():
        st.error(f"Schema file not found: {schema_path}")
        return None
    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        # st.info(f"Successfully loaded schema: {schema_path.name}") # Reduce noise
        return schema
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON in schema file {schema_path}: {e}")
        return None
    except Exception as e:
        st.error(f"Failed to read schema file {schema_path}: {e}")
        return None

def describe_property(prop_name: str, prop_schema: dict, required_fields: List[str], indent_level=0) -> str:
    """Recursively describes a schema property for the prompt."""
    description = ""
    indent = "  " * indent_level
    constraints = []
    prop_description_raw = prop_schema.get("description", "")
    prop_description = str(prop_description_raw) if prop_description_raw is not None else ""

    if prop_description:
        description += f"{indent}# {prop_description}\n"

    prop_type = prop_schema.get("type")
    if prop_type: constraints.append(f"type: {prop_type}")

    if prop_type == "string":
        if "minLength" in prop_schema: constraints.append(f"min_len: {prop_schema['minLength']}")
        if "maxLength" in prop_schema: constraints.append(f"max_len: {prop_schema['maxLength']}")
        if "pattern" in prop_schema: constraints.append(f"pattern: `{prop_schema['pattern']}`")
        if "enum" in prop_schema: constraints.append(f"enum: {prop_schema['enum']}")
        if "format" in prop_schema: constraints.append(f"format: {prop_schema['format']}")
    elif prop_type in ["number", "integer"]:
        if "minimum" in prop_schema: constraints.append(f"min: {prop_schema['minimum']}")
        if "maximum" in prop_schema: constraints.append(f"max: {prop_schema['maximum']}")
    elif prop_type == "array":
        if "minItems" in prop_schema: constraints.append(f"min_items: {prop_schema['minItems']}")
        if "maxItems" in prop_schema: constraints.append(f"max_items: {prop_schema['maxItems']}")

    constraint_str = ", ".join(constraints) if constraints else "no specific constraints"
    required_marker = "**(required)**" if prop_name in required_fields else "(optional)"
    description += f"{indent}- `{prop_name}` {required_marker} ({constraint_str})\n"

    if prop_type == "object" and "properties" in prop_schema:
        nested_required = prop_schema.get("required", [])
        obj_description_raw = prop_schema.get("description", "")
        obj_description = str(obj_description_raw) if obj_description_raw is not None else ""
        if obj_description and obj_description not in description:
             description += f"{indent}  # {obj_description}\n"
        description += f"{indent}  Properties:\n"
        for nested_name, nested_schema in prop_schema["properties"].items():
            description += describe_property(nested_name, nested_schema, nested_required, indent_level + 1)

    elif prop_type == "array" and "items" in prop_schema:
        item_schema = prop_schema["items"]
        item_type = item_schema.get("type", "any")
        arr_description_raw = prop_schema.get("description", "")
        arr_description = str(arr_description_raw) if arr_description_raw is not None else ""
        if arr_description and arr_description not in description:
            description += f"{indent}  # {arr_description}\n"
        description += f"{indent}  Each item should be type '{item_type}'.\n"

        if item_type == "object" and "properties" in item_schema:
             item_obj_description_raw = item_schema.get("description", "Each item object contains:")
             item_obj_description = str(item_obj_description_raw) if item_obj_description_raw is not None else ""
             description += f"{indent}  # {item_obj_description}\n"
             nested_required = item_schema.get("required", [])
             for item_prop_name, item_prop_schema in item_schema["properties"].items():
                 description += describe_property(item_prop_name, item_prop_schema, nested_required, indent_level + 2)

    return description

def generate_comprehensive_prompt(schema: dict, doc_type: str, ocr_text: str) -> str:
    """Generates ONE detailed prompt for the LLM to structure OCR text."""
    if not schema:
        raise ValueError("Schema cannot be None for prompt generation.")
    doc_type_name = doc_type.replace('_', ' ').title()
    schema_title = schema.get("title", f"'{doc_type_name}' document")
    schema_description_raw = schema.get("description", "")
    schema_description = str(schema_description_raw) if schema_description_raw is not None else ""

    prompt_intro_parts = [
        f"You are an expert document processing assistant specialized in extracting structured information from **TEXT** with **maximum accuracy**, specifically focusing on **{doc_type_name}** documents.",
        f"Analyze the following text meticulously. It was extracted via OCR from a {doc_type_name}.",
        f"Your task is to extract information based *only* on this text and structure it precisely according to the following JSON schema definition for a **{schema_title}**.",
    ]
    if schema_description:
        prompt_intro_parts.append(f"Schema description: {schema_description}")

    prompt_intro_parts.extend([
        f"Extract values for the fields below directly from the provided text. Adhere **strictly** to all constraints:",
        f"  - **Data Types** (string, number, boolean, array, object).",
        f"  - **Required Fields** (marked **(required)**) - these MUST be included in the JSON.",
        f"  - **String Constraints** (length, patterns like IDs/Dates/PAN, enums, format). **CRITICAL: Match patterns exactly.** Convert dates to 'YYYY-MM-DD' if possible based on text.",
        f"  - **Numeric/Array Constraints** (min/max values or items).",
        f"  - **Nested Structures**: Follow the object/array structure precisely."
    ])
    prompt_intro = "\n".join(prompt_intro_parts) + "\n"

    field_descriptions = "\n--- JSON Schema Structure & Fields ---\n"
    top_level_properties = schema.get("properties", {})
    top_level_required = schema.get("required", [])
    if not top_level_properties:
        raise ValueError("Schema missing top-level 'properties'.")

    if schema_description and "Schema description:" not in prompt_intro:
        field_descriptions += f"# {schema_description}\n"

    for prop_name, prop_schema in top_level_properties.items():
        field_descriptions += describe_property(prop_name, prop_schema, top_level_required, 0)

    ocr_text_section = f"\n--- OCR Text Content ---\n```text\n{ocr_text}\n```\n--- End OCR Text Content ---"

    prompt_outro = (
        "\n--- Output Instructions ---\n"
        "1.  **Output ONLY the JSON:** Your *entire* response MUST be a single, valid JSON object. Start with `{` and end with `}`. NO explanations, apologies, summaries, or markdown backticks (```json ... ```) surrounding the JSON object itself."
        "2.  **Strict Schema Adherence:** The output JSON MUST conform strictly to the structure, types, and constraints defined in the schema above, using only information present in the OCR text.",
        "3.  **Required Fields:** Ensure all fields marked **(required)** are present in the output JSON. If a required field's value absolutely cannot be found in the text, use JSON `null`, but only if the schema allows `null` for that field type (check `type: ['string', 'null']`). If `null` is not allowed, you must infer a plausible value if possible or indicate the failure clearly if extraction is impossible (though try to avoid this by finding *something* in the text).",
        "4.  **Optional Fields:** If an optional field's value is not found in the text, **OMIT the field entirely** from the JSON output. DO NOT use `null` for missing optional fields unless the schema explicitly allows `null` (e.g., `type: ['string', 'null']`).",
        "5.  **Accuracy Paramount:** Extract values accurately. Double-check numbers, IDs (like PAN), names, and dates against the OCR text. Ensure correct formatting (e.g., `YYYY-MM-DD` for dates if specified).",
        "6.  **No Extra Information:** Do not add fields not defined in the schema. Do not include comments within the JSON unless the schema specifies it (which is rare)."
        "7.  **Currency/Units:** Extract numeric values for currency fields, generally omitting symbols like '₹' or '$' unless the schema pattern requires them. Check the specific field description."
        "8.  **Boolean Fields**: Interpret textual cues (like 'Yes'/'No', 'True'/'False', presence/absence of a checkmark description) to determine boolean `true` or `false` values."
    )
    return str(prompt_intro) + str(field_descriptions) + str(ocr_text_section) + str(prompt_outro)

def post_process_extracted_data(data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """Applies rule-based cleaning to extracted data *before* validation."""
    processed_data = {}
    if not isinstance(data, dict):
        # Log this potentially?
        # print(f"⚠️ Post-processing warning: Input data is not a dictionary ({type(data)}), returning as is.")
        return data # Return original if not a dict
    if not schema or 'properties' not in schema:
        # print(f"⚠️ Post-processing warning: Schema invalid or missing properties, returning data as is.")
        return data # Return original if schema is invalid

    schema_props = schema.get("properties", {})

    for key, value in data.items():
        if key not in schema_props: # Skip keys not in schema
             continue

        prop_schema = schema_props.get(key, {})
        prop_type = prop_schema.get("type") # Can be string or list
        original_value = value
        cleaned_value = value

        # Handle null values explicitly - keep them if they are present
        if cleaned_value is None:
            processed_data[key] = None
            continue

        # --- Apply Specific Cleaning Rules ---
        # Generic String Cleanup
        if isinstance(cleaned_value, str):
            cleaned_value = cleaned_value.strip()
            # cleaned_value = re.sub(r'\s+', ' ', cleaned_value).strip() # Optional: Normalize whitespace

        # Rule for PAN Number (adjust key if schema differs)
        # Example: assumes key is 'pan_number' in schema
        if key == "pan_number" and isinstance(cleaned_value, str) and "string" in prop_type:
            cleaned_value = re.sub(r'[^A-Z0-9]', '', cleaned_value.upper())
            # Optional: Check length if defined in schema
            # if "maxLength" in prop_schema and len(cleaned_value) > prop_schema["maxLength"]:
            #     cleaned_value = cleaned_value[:prop_schema["maxLength"]] # Truncate? Or log warning?

        # Rule for Currency fields (example: net_salary)
        # Example: assumes key is 'net_salary' and type is 'number' or 'integer'
        elif key == "net_salary" and isinstance(cleaned_value, str) and ("number" in prop_type or "integer" in prop_type):
            # Remove currency symbols, commas, spaces
            cleaned_value_numeric = re.sub(r'[₹$,€£\s]', '', cleaned_value)
            # Check if it looks like a valid number after cleaning
            if re.match(r'^-?\d+(\.\d+)?$', cleaned_value_numeric):
                # Try converting to float or int based on schema type
                try:
                    if "integer" in prop_type:
                        cleaned_value = int(float(cleaned_value_numeric)) # Convert via float first for ".00" cases
                    else: # Assume number (float)
                        cleaned_value = float(cleaned_value_numeric)
                except ValueError:
                     cleaned_value = original_value # Revert if conversion fails
            else:
                # If it doesn't look numeric after cleaning, maybe keep original string? Or set to None?
                cleaned_value = original_value # Keep original potentially non-numeric string? Or maybe None? Needs decision.

        # Rule for Date Fields (example: 'YYYY-MM-DD' format)
        elif isinstance(cleaned_value, str) and "string" in prop_type and prop_schema.get("format") == "date":
             # Add more flexible date parsing if needed (e.g., using dateutil.parser)
             match_dmy_slash = re.match(r'(\d{1,2})\s?[/-]\s?(\d{1,2})\s?[/-]\s?(\d{4})', cleaned_value) # dd/mm/yyyy or dd-mm-yyyy
             match_mdy_slash = re.match(r'(\d{1,2})\s?[/-]\s?(\d{1,2})\s?[/-]\s?(\d{4})', cleaned_value) # mm/dd/yyyy or mm-dd-yyyy (needs context)
             match_ymd_dash = re.match(r'(\d{4})\s?[/-]\s?(\d{1,2})\s?[/-]\s?(\d{1,2})', cleaned_value) # yyyy-mm-dd
             match_mon_d_y = re.match(r'([A-Za-z]{3,})\s+(\d{1,2}),?\s+(\d{4})', cleaned_value) # Jan 1, 2024

             parsed_date = None
             try:
                if match_ymd_dash:
                    y, m, d = match_ymd_dash.groups()
                    parsed_date = f"{y}-{int(m):02d}-{int(d):02d}"
                elif match_dmy_slash: # Assuming DD/MM/YYYY common in some regions
                    d, m, y = match_dmy_slash.groups()
                    parsed_date = f"{y}-{int(m):02d}-{int(d):02d}"
                # Add more parsing logic here (e.g., using dateutil.parser for flexibility)
                # from dateutil import parser
                # try:
                #     parsed_dt_obj = parser.parse(cleaned_value)
                #     parsed_date = parsed_dt_obj.strftime('%Y-%m-%d')
                # except ValueError:
                #     pass # Failed to parse

                if parsed_date:
                    # Final check if it matches YYYY-MM-DD
                    if re.match(r'^\d{4}-\d{2}-\d{2}$', parsed_date):
                         cleaned_value = parsed_date
                    # else: keep original if parsing resulted in wrong format somehow

             except Exception:
                 pass # Keep original value if parsing fails

        # Recursive processing for nested objects and arrays
        elif isinstance(cleaned_value, dict) and "object" in prop_type:
            # Make sure the prop_schema itself represents the object's schema
            cleaned_value = post_process_extracted_data(cleaned_value, prop_schema)

        elif isinstance(cleaned_value, list) and "array" in prop_type:
            item_schema = prop_schema.get("items")
            if item_schema:
                cleaned_array = []
                item_type = item_schema.get("type", "any")
                for item in cleaned_value:
                    if isinstance(item, dict) and item_type == "object":
                        cleaned_array.append(post_process_extracted_data(item, item_schema))
                    elif isinstance(item, str) and item_type == "string":
                         cleaned_array.append(item.strip())
                         # Add other simple type cleaning if needed
                    else:
                        cleaned_array.append(item) # Keep item as is
                cleaned_value = cleaned_array

        # Log if changed (optional, can be verbose)
        # if cleaned_value != original_value:
        #      print(f"   Post-processing '{key}': '{original_value}' -> '{cleaned_value}'")

        processed_data[key] = cleaned_value

    # Ensure all required fields from schema are present, add null if allowed and missing
    required_fields = schema.get("required", [])
    for req_key in required_fields:
        if req_key not in processed_data:
            req_prop_schema = schema_props.get(req_key, {})
            req_prop_type = req_prop_schema.get("type", [])
            if isinstance(req_prop_type, list) and "null" in req_prop_type:
                 processed_data[req_key] = None # Add null if missing and allowed
            # else: it will fail validation later, which is intended

    return processed_data


# --- Streamlit App ---
st.set_page_config(page_title="Document AI Processor", layout="wide")

st.title("📄 Document AI Processor")
st.markdown("Upload a document image, select its type, and extract structured data.")

# --- State Initialization ---
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = None
if "raw_llm_output" not in st.session_state:
    st.session_state.raw_llm_output = None
if "processed_json" not in st.session_state:
    st.session_state.processed_json = None
if "validation_status" not in st.session_state:
    st.session_state.validation_status = None # Tuple (bool, message)
if "error_message" not in st.session_state:
    st.session_state.error_message = None
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None # Store the PIL image object
if "doc_type" not in st.session_state:
     st.session_state.doc_type = DEFAULT_DOC_TYPE # Initialize with default

# --- Model Initialization (Cached) ---
@st.cache_resource(show_spinner=False) # Hide default spinner, use st.status
def initialize_models():
    """Loads OCR engine, LLM, and Tokenizer."""
    models = {"ocr": None, "llm": None, "tokenizer": None}
    init_errors = []

    # 1. Initialize PaddleOCR
    ocr_init_time = time.time()
    if paddleocr_imported:
        try:
            st.write("Initializing PaddleOCR Engine (CPU)...")
            # Use lower logging level for paddleocr if possible
            models["ocr"] = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False, logging_level='ERROR')
            st.write(f"✅ PaddleOCR initialized ({time.time() - ocr_init_time:.2f}s)")
        except Exception as e:
            error_msg = f"PaddleOCR Initialization Error: {e}"
            st.error(error_msg)
            init_errors.append(error_msg)
            models["ocr"] = None # Ensure it's None on error
    else:
        error_msg = "OCR Skipped: 'paddleocr' library not imported or failed to import."
        st.warning(error_msg)
        init_errors.append(error_msg)

    # 2. Initialize LLM and Tokenizer
    llm_init_time = time.time()
    try:
        st.write(f"Initializing Tokenizer for '{MODEL_ID}'...")
        # Consider trust_remote_code=True if required by the model card
        models["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN) #, trust_remote_code=True)

        st.write(f"Initializing LLM '{MODEL_ID}' (this might take time)...")

        # Check for GPU availability for quantization
        quantization_config = None
        torch_dtype = torch.float16 # Default for GPU
        if torch.cuda.is_available():
             st.write("CUDA available. Checking for BitsAndBytes quantization...")
             try:
                 # 4-bit quantization for memory saving
                 quantization_config = BitsAndBytesConfig(
                     load_in_4bit=True,
                     bnb_4bit_compute_dtype=torch.float16, # Or bfloat16 if supported
                     bnb_4bit_quant_type="nf4",
                     bnb_4bit_use_double_quant=True,
                 )
                 st.write("Using 4-bit quantization (BitsAndBytes).")
             except ImportError:
                  st.warning("BitsAndBytes not installed (`pip install bitsandbytes`). Quantization skipped.")
             except Exception as e:
                  st.warning(f"Failed to create BitsAndBytes config: {e}. Quantization skipped.")
        else:
            st.write("CUDA not available. Using CPU (might be slow).")
            torch_dtype = torch.float32 # Use float32 on CPU for better compatibility

        try:
             models["llm"] = AutoModelForCausalLM.from_pretrained(
                 MODEL_ID,
                 torch_dtype=torch_dtype,
                 device_map="auto", # Let accelerate handle device placement
                 token=HF_TOKEN,
                 quantization_config=quantization_config, # Apply if defined
                 # trust_remote_code=True, # Add if model requires it
                 low_cpu_mem_usage=True if torch.cuda.is_available() else False # More relevant for large models on GPU load
             )
             try:
                 # Check device placement if possible
                 model_device = next(models["llm"].parameters()).device
                 st.write(f"✅ LLM loaded on device(s). Primary device: {model_device}")
             except Exception:
                  st.write(f"✅ LLM loaded successfully via device_map='auto'.")

        except ImportError as e_accel:
             # Fallback if accelerate is not installed or device_map fails
             st.warning(f"'accelerate' library issue ({e_accel}). Falling back to manual device placement.")
             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
             st.write(f"Placing model manually on: {device}")
             # Load without device_map, manually move later
             models["llm"] = AutoModelForCausalLM.from_pretrained(
                 MODEL_ID,
                 torch_dtype=torch_dtype,
                 token=HF_TOKEN,
                 quantization_config=quantization_config, # Apply if defined
                 # trust_remote_code=True,
                 low_cpu_mem_usage=True # Helps when loading large models
             ).to(device)
             st.write(f"✅ LLM loaded successfully on: {models['llm'].device}")

        st.write(f"✅ Tokenizer & LLM initialized ({time.time() - llm_init_time:.2f}s)")

    except Exception as e:
        error_msg = f"LLM/Tokenizer Initialization Error: {e}"
        st.error(error_msg)
        init_errors.append(error_msg)
        models["llm"] = None
        models["tokenizer"] = None

    return models, init_errors


# --- Load Models ---
with st.spinner("Initializing AI models..."):
     # This runs only once per session due to @st.cache_resource
     loaded_models, initialization_errors = initialize_models()

ocr_engine = loaded_models["ocr"]
llm_model = loaded_models["llm"]
llm_tokenizer = loaded_models["tokenizer"]
init_error_message = "\n".join(initialization_errors) if initialization_errors else None


# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("⚙️ Inputs")

    # Document Type Selection
    doc_type_options = list(SCHEMA_MAP.keys())
    # Use session state value for selectbox, update state on change
    st.session_state.doc_type = st.selectbox(
        "1. Select Document Type:",
        options=doc_type_options,
        index=doc_type_options.index(st.session_state.doc_type) if st.session_state.doc_type in doc_type_options else 0,
        key="doc_type_selector" # Unique key
    )

    # Image Upload
    uploaded_file = st.file_uploader("2. Upload Document Image:", type=["png", "jpg", "jpeg"])

    # Disable button if models failed to load or no file is uploaded
    process_button_disabled = not uploaded_file or not ocr_engine or not llm_model or not llm_tokenizer
    process_button = st.button("🚀 Process Document", type="primary", disabled=process_button_disabled)

    st.divider()
    if process_button_disabled and init_error_message:
         st.warning(f"Processing disabled due to initialization errors:\n{init_error_message}")
    elif process_button_disabled and not uploaded_file:
         st.caption("Please upload an image.")
    elif process_button_disabled:
         st.caption("Models not loaded correctly. Check console logs.")
    st.caption("Ensure the correct document type is selected before processing.")

# --- Main Processing Logic ---
if process_button and uploaded_file and not process_button_disabled:
    # Reset state for new processing run
    st.session_state.processing_complete = False
    st.session_state.ocr_text = None
    st.session_state.raw_llm_output = None
    st.session_state.processed_json = None
    st.session_state.validation_status = None
    st.session_state.error_message = None
    st.session_state.uploaded_image = None

    # Get selected document type from state
    selected_doc_type = st.session_state.doc_type
    schema_path = SCHEMA_MAP.get(selected_doc_type)

    # Check and Load Schema first
    if not schema_path or not schema_path.exists():
        error_msg = f"Schema file not found for '{selected_doc_type}' at: {schema_path}"
        st.error(error_msg)
        st.session_state.error_message = error_msg
        st.stop() # Stop execution if schema is missing

    schema = load_schema(schema_path)
    if not schema:
        error_msg = f"Failed to load or parse schema for '{selected_doc_type}'. Check schema file format and JSON validity."
        st.error(error_msg) # load_schema might have already shown an error
        st.session_state.error_message = error_msg
        st.stop() # Stop if schema loading fails

    # Process Image
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.session_state.uploaded_image = image # Store the PIL image

        # Use BytesIO for in-memory handling
        img_byte_arr = io.BytesIO()
        # Save in a format PaddleOCR commonly supports
        image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()

        # --- Stage 1: OCR ---
        ocr_text = None
        ocr_error = None
        with st.status("1/4: Performing OCR...", expanded=True) as status:
            st.write("Extracting text from the image...")
            ocr_start_time = time.time()
            try:
                # Pass image bytes directly to PaddleOCR's ocr method
                # Note: Some versions/setups might still prefer a file path.
                # If bytes fail, fallback to saving a temp file.
                ocr_raw_output = ocr_engine.ocr(img_bytes, cls=True)
                elapsed_ocr_time = time.time() - ocr_start_time

                if ocr_raw_output and isinstance(ocr_raw_output, list) and len(ocr_raw_output) > 0:
                    ocr_results = ocr_raw_output[0] # Result is usually nested [[...]]
                    if ocr_results and isinstance(ocr_results, list):
                        lines = [line[1][0] for line in ocr_results if line and len(line) > 1 and isinstance(line[1], tuple) and len(line[1]) > 0]
                        ocr_text = "\n".join(lines)
                        if ocr_text:
                             st.write(f"OCR successful ({len(lines)} lines extracted). Time: {elapsed_ocr_time:.2f}s")
                             st.session_state.ocr_text = ocr_text
                             status.update(label="OCR Completed Successfully!", state="complete", expanded=False)
                        else:
                            ocr_error = "OCR completed but no text was extracted."
                            st.warning(ocr_error)
                            status.update(label=ocr_error, state="warning", expanded=False)
                    else:
                         ocr_error = "OCR ran but returned empty or unexpected results."
                         st.warning(ocr_error)
                         status.update(label=ocr_error, state="warning", expanded=False)
                else:
                     ocr_error = "OCR engine returned empty or invalid output."
                     st.warning(ocr_error)
                     status.update(label=ocr_error, state="warning", expanded=False)

            except Exception as e:
                ocr_error = f"Error during OCR processing: {e}"
                st.error(ocr_error)
                status.update(label=f"OCR Error: {e}", state="error", expanded=True)
                st.session_state.error_message = ocr_error
                st.stop() # Stop processing if OCR fails critically

        # --- Stage 2: LLM Structuring ---
        raw_llm_output = None
        structure_error = None
        if st.session_state.ocr_text: # Only proceed if OCR text exists
            with st.status("2/4: Structuring Text with LLM...", expanded=True) as status:
                st.write(f"Generating prompt for '{selected_doc_type}'...")
                try:
                    prompt = generate_comprehensive_prompt(schema, selected_doc_type, st.session_state.ocr_text)
                except Exception as e:
                    structure_error = f"Failed to generate LLM prompt: {e}"
                    st.error(structure_error)
                    status.update(label=structure_error, state="error", expanded=True)
                    st.session_state.error_message = structure_error
                    st.stop()

                st.write(f"Sending prompt to LLM ({MODEL_ID})...")
                llm_start_time = time.time()
                try:
                    inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=llm_tokenizer.model_max_length - MAX_TOKENS).to(llm_model.device) # Reserve space for generation

                    generation_kwargs = {
                        "max_new_tokens": MAX_TOKENS,
                        "do_sample": False, # Use greedy decoding for deterministic JSON
                        "temperature": 0.0,
                        "pad_token_id": llm_tokenizer.eos_token_id,
                        "eos_token_id": llm_tokenizer.eos_token_id
                    }

                    with torch.inference_mode():
                        generate_ids = llm_model.generate(**inputs, **generation_kwargs)

                    # Decode only the newly generated tokens
                    output_ids = generate_ids[0][inputs['input_ids'].shape[1]:]
                    raw_llm_output = llm_tokenizer.decode(output_ids, skip_special_tokens=True).strip()
                    elapsed_llm_time = time.time() - llm_start_time

                    if raw_llm_output:
                        st.session_state.raw_llm_output = raw_llm_output
                        st.write(f"LLM structuring successful. Time: {elapsed_llm_time:.2f}s")
                        status.update(label="LLM Structuring Completed!", state="complete", expanded=False)
                    else:
                        structure_error = "LLM returned an empty response."
                        st.warning(structure_error)
                        status.update(label=structure_error, state="warning", expanded=False)

                except Exception as e:
                    structure_error = f"LLM Generation Error: {e}"
                    st.error(structure_error)
                    status.update(label=structure_error, state="error", expanded=True)
                    st.session_state.error_message = structure_error
                    # Don't necessarily stop here, maybe parsing can still work if raw_llm_output has partial data? Or stop if needed.
                    # st.stop()
        else:
            st.warning("Skipping LLM structuring because no text was extracted by OCR.")
            # Ensure subsequent steps know this stage was skipped
            st.session_state.raw_llm_output = None


        # --- Stage 3: Parsing & Cleaning ---
        processed_json = None
        parse_clean_error = None
        if st.session_state.raw_llm_output: # Only proceed if LLM output exists
            with st.status("3/4: Parsing & Cleaning Data...", expanded=True) as status:
                st.write("Attempting to parse JSON from LLM output...")
                json_string = None
                parsed_json_intermediate = None
                try:
                    # Robust JSON extraction from raw output
                    # 1. Look for ```json ... ``` block
                    match_md = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", st.session_state.raw_llm_output, re.DOTALL | re.IGNORECASE)
                    if match_md:
                        json_string = match_md.group(1)
                        st.write("   Found JSON block inside markdown.")
                    else:
                        # 2. Look for first '{' and last '}'
                        first_brace = st.session_state.raw_llm_output.find('{')
                        last_brace = st.session_state.raw_llm_output.rfind('}')
                        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                            json_string = st.session_state.raw_llm_output[first_brace : last_brace + 1]
                            st.write("   Found JSON block by locating first '{' and last '}'.")
                        else:
                            parse_clean_error = "Parsing Error: Could not locate a JSON object ({...}) in the LLM output."
                            st.warning(parse_clean_error)
                            # Don't immediately stop, cleaning might still run on partial data if needed?
                            # status.update(...) state="warning" ?

                    if json_string:
                        parsed_json_intermediate = json.loads(json_string)
                        st.write("   Successfully parsed JSON.")

                        st.write("Applying post-processing rules...")
                        # Apply cleaning rules
                        processed_json = post_process_extracted_data(parsed_json_intermediate, schema)
                        st.session_state.processed_json = processed_json
                        st.write("   Post-processing complete.")
                        status.update(label="Parsing & Cleaning Done!", state="complete", expanded=False)

                    elif not parse_clean_error: # If braces weren't found, set error
                         parse_clean_error = "Parsing Error: No JSON content identified in LLM output."
                         st.warning(parse_clean_error)
                         status.update(label=parse_clean_error, state="warning", expanded=False)

                except json.JSONDecodeError as e:
                    parse_clean_error = f"JSON Parsing Error: {e}. Check the 'Raw LLM Output' below."
                    st.error(parse_clean_error)
                    status.update(label=parse_clean_error, state="error", expanded=True)
                    st.session_state.error_message = parse_clean_error # Store critical parsing error
                    # Keep raw output for debugging, but stop further processing
                except Exception as e:
                    parse_clean_error = f"Data Processing Error: {e}"
                    st.error(parse_clean_error)
                    status.update(label=parse_clean_error, state="error", expanded=True)
                    st.session_state.error_message = parse_clean_error
        else:
            st.warning("Skipping Parsing & Cleaning because no raw output was generated by the LLM.")


        # --- Stage 4: Validation ---
        validation_status = None
        if st.session_state.processed_json is not None: # Only validate if parsing/cleaning succeeded
            with st.status("4/4: Validating Data Against Schema...", expanded=True) as status:
                st.write(f"Validating the extracted JSON against the '{selected_doc_type}' schema...")
                if not validator_imported:
                    st.warning("Validation skipped: Validator module could not be imported.")
                    validation_msg = "Validation skipped: Validator module import failed."
                    is_valid = False # Treat as invalid if validator missing
                    status.update(label="Validation Skipped (Import Error)", state="warning", expanded=False)
                elif not isinstance(st.session_state.processed_json, dict):
                     validation_msg = f"Validation skipped: Processed data is not a dictionary (type: {type(st.session_state.processed_json).__name__})."
                     st.warning(validation_msg)
                     is_valid = False
                     status.update(label="Validation Skipped (Invalid Data Type)", state="warning", expanded=False)
                else:
                    try:
                        is_valid, validation_msg = validate_extraction(
                            data=st.session_state.processed_json,
                            # Pass doc_type only if your validator function needs it
                            document_type=selected_doc_type
                        )
                        if is_valid:
                            st.write("Validation successful.")
                            status.update(label="Validation Passed!", state="complete", expanded=False)
                        else:
                            st.write("Validation failed.")
                            # The message already contains details
                            status.update(label="Validation Failed (See Results)", state="error", expanded=False)
                    except Exception as e:
                         is_valid = False
                         validation_msg = f"Unexpected Validation Error: {e}"
                         st.error(validation_msg)
                         status.update(label=f"Validation Error: {e}", state="error", expanded=True)

                st.session_state.validation_status = (is_valid, validation_msg)

        elif parse_clean_error:
             st.warning(f"Skipping Validation due to Parsing/Cleaning Error: {parse_clean_error}")
             st.session_state.validation_status = (False, f"Validation skipped due to parsing/cleaning error: {parse_clean_error}")
        else:
            st.warning("Skipping Validation because no processed JSON data is available.")
            st.session_state.validation_status = (False, "Validation skipped: No processed JSON data available.")


        # --- Mark processing as complete ---
        st.session_state.processing_complete = True

    except Exception as e:
        # Catch-all for unexpected errors during the main processing flow
        st.session_state.error_message = f"An unexpected error occurred during processing: {e}"
        st.error(st.session_state.error_message)
        import traceback
        st.error(f"Traceback:\n{traceback.format_exc()}") # Show traceback in Streamlit for debugging
        st.session_state.processing_complete = False # Ensure it's marked as not complete


# --- Display Results ---
st.divider()
st.header("📊 Results")

# Condition to show results: Processing finished OR an error stopped it mid-way
show_results = st.session_state.processing_complete or st.session_state.error_message

if show_results:
    # Use columns for a side-by-side view
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Document")
        if st.session_state.uploaded_image:
            st.image(st.session_state.uploaded_image, caption="Uploaded Image", use_column_width=True)
        elif uploaded_file: # If processing failed before image could be stored
             try:
                  # Try reloading image if needed, handle potential errors
                  reloaded_image = Image.open(uploaded_file).convert("RGB")
                  st.image(reloaded_image, caption="Uploaded Image (Processing Halted)", use_column_width=True)
             except Exception as img_err:
                  st.warning(f"Could not reload uploaded image for display after error: {img_err}")
        else:
            st.info("Image was not processed or display failed.")

    with col2:
        st.subheader("Extracted & Validated Data")

        # Display Validation Status First
        if st.session_state.validation_status:
            is_valid, validation_msg = st.session_state.validation_status
            if is_valid:
                st.success(f"✅ Schema Validation Passed!")
                # Show success message only if it's informative
                if validation_msg and validation_msg.lower() not in ["validation successful.", "validation successful"]:
                    st.info(validation_msg)
            else:
                # Display main error message
                st.error(f"❌ Schema Validation Failed / Skipped")
                # Show detailed validation errors/messages in a text area
                st.text_area("Validation Details:", value=validation_msg, height=100, disabled=True, key="validation_details")
        else:
            # Handles case where validation didn't run at all
             if not st.session_state.error_message: # If no other major error occurred
                 st.warning("Validation was not performed or status is unavailable.")
             # If there was an earlier error, validation status is less important than the root cause

        # Display the Final JSON
        if st.session_state.processed_json is not None:
            st.caption("Final Processed JSON:")
            st.json(st.session_state.processed_json, expanded=True) # Expand by default
        else:
            # Explain why JSON isn't available
            if st.session_state.error_message and "Parsing" in st.session_state.error_message:
                 st.warning("Final JSON not available due to a parsing error.")
            elif st.session_state.error_message and "LLM" in st.session_state.error_message:
                 st.warning("Final JSON not available due to an LLM generation error.")
            elif not st.session_state.raw_llm_output and st.session_state.ocr_text :
                 st.warning("Final JSON not available because the LLM did not return output.")
            elif not st.session_state.ocr_text:
                 st.warning("Final JSON not available because OCR did not detect text.")
            else:
                 st.info("No final JSON data was successfully processed.")


    # Expanders for Raw Data (Always show if available, helps debugging)
    st.divider()
    st.subheader("Raw Data & Intermediate Steps")
    with st.expander("Raw OCR Text", expanded=False):
        if st.session_state.ocr_text:
            st.text_area("OCR Output:", value=st.session_state.ocr_text, height=200, disabled=True, key="ocr_output_area")
        else:
            st.caption("No OCR text was extracted or available.")

    with st.expander("Raw LLM Output (Before Parsing/Cleaning)", expanded=False):
        if st.session_state.raw_llm_output:
            # Display as code block
            st.code(st.session_state.raw_llm_output, language='text', line_numbers=True)
        else:
            st.caption("No raw LLM output was generated or available.")

    # Display final critical error if one occurred during the run
    if st.session_state.error_message and not st.session_state.validation_status: # Show root error if validation didn't even run
         st.error(f"**Processing Halted Error:** {st.session_state.error_message}")
    elif st.session_state.error_message and st.session_state.validation_status and not st.session_state.validation_status[0]: # Show if validation ran but failed/skipped
        st.warning(f"Note: An error occurred during processing: {st.session_state.error_message}")


elif not uploaded_file:
    st.info("Please upload a document image using the sidebar to begin.")
elif init_error_message:
     st.error(f"Cannot proceed due to initialization error(s):\n{init_error_message}")


# --- Footer ---
st.markdown("---")
st.caption("Document AI Pipeline using PaddleOCR and Hugging Face Transformers.")