'''import json
from jsonschema import validate, ValidationError

def validate_extraction(data: dict, schema_file: str):
    with open(schema_file, "r") as f:
        schema = json.load(f)
    try:
        validate(instance=data, schema=schema)
        return True, "✅ All fields valid."
    except ValidationError as e:
        return False, f"❌ Validation failed: {e.message}"
'''
# src/schema_validator.py
import json
import os
from pathlib import Path
from jsonschema import validate, ValidationError

# Define the base directory assuming 'src' is one level down from the project root
# where the 'schemas' directory also resides.
try:
    # This works when the script is run directly or imported from the notebook in the root
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
     # Fallback for environments where __file__ might not be defined (like some interactive sessions)
     # Assumes the current working directory is the project root
    BASE_DIR = Path.cwd()


# Mapping of document types (matching notebook's DOC_TYPE variable)
# to their schema file paths relative to the BASE_DIR.
# src/schema_validator.py

# ... (BASE_DIR definition) ...

# Mapping of document types (matching notebook's DOC_TYPE variable)
SCHEMA_MAP = {
    "loan_application": BASE_DIR / "schemas/loan_application.json",
    "property": BASE_DIR / "schemas/property.json",
    "bank_statements": BASE_DIR / "schemas/bank_statements.json", # Make sure this one is correct too (plural?)
    "income_salary_slip": BASE_DIR / "schemas/income_salary_slip.json", # <--- CHANGE THIS KEY AND FILENAME
    # Add other document types and their corresponding schema paths here
}

# ... (rest of the validator code) ...

def validate_extraction(data: dict, document_type: str):
    """
    Validates extracted data against the schema corresponding to the document type.

    Args:
        data: The dictionary containing the extracted data (parsed JSON).
        document_type: A string identifying the document type (e.g., 'loan_application').
                       This must be a key in SCHEMA_MAP.

    Returns:
        A tuple (bool, str):
            (True, success_message) if validation passes.
            (False, error_message) if validation fails, schema is missing, or other error occurs.
    """
    schema_path = SCHEMA_MAP.get(document_type)

    if not schema_path:
        valid_types = ", ".join(SCHEMA_MAP.keys())
        return False, f"❌ Validation Error: Unsupported document type provided: '{document_type}'. Valid types are: {valid_types}."

    if not isinstance(schema_path, Path):
         # Should not happen with the current setup, but good practice
         schema_path = Path(schema_path)

    if not schema_path.is_file():
        # Try to provide a helpful path relative to project root if possible
        try:
            relative_path = schema_path.relative_to(BASE_DIR)
        except ValueError:
             relative_path = schema_path # Show absolute if not relative to base

        return False, f"❌ Validation Error: Schema file not found at expected path: '{relative_path}' (Full resolved path: {schema_path})"

    try:
        with open(schema_path, "r", encoding='utf-8') as f: # Specify encoding
            schema = json.load(f)
    except json.JSONDecodeError as e:
            return False, f"❌ Validation Error: Invalid JSON in schema file '{schema_path.name}': {e}"
    except Exception as e:
        return False, f"❌ Validation Error: Could not read schema file '{schema_path.name}': {e}"

    if not isinstance(data, dict):
        return False, f"❌ Validation Error: Input data is not a dictionary (type: {type(data).__name__}). Cannot validate."

    try:
        validate(instance=data, schema=schema)
        return True, f"✅ Schema Validated: Extracted data conforms to the '{document_type}' schema ({schema_path.name})."
    except ValidationError as e:
        # Provide more context from the error object
        error_field = "'"+"' -> '".join(map(str, e.path)) + "'" if e.path else "Top Level"
        # Attempt to show the problematic value if it's not too large
        instance_snippet = str(e.instance)
        if len(instance_snippet) > 100:
            instance_snippet = instance_snippet[:97] + "..."

        return False, (f"❌ Schema Validation Failed for field {error_field}: {e.message}. "
                      f"Problematic value: '{instance_snippet}'. (Schema: '{schema_path.name}')")
    except Exception as e:
        # Catch other potential errors during validation itself
        return False, f"❌ Unexpected Validation Error during jsonschema.validate: {e} (Schema: '{schema_path.name}')"