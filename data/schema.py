"""
Extraction schemas for DocVL-IN.

Every document image is paired with a ground-truth JSON object conforming to one of
these schemas. The model is trained to emit exactly this JSON given the image + a
task prompt naming the document type. Keeping schemas explicit (rather than free-form
key-value extraction) makes evaluation (field-level F1 / exact match) well-defined.
"""

from __future__ import annotations
from typing import Optional, Literal
from pydantic import BaseModel, Field


class InvoiceFields(BaseModel):
    doc_type: Literal["invoice"] = "invoice"
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = Field(None, description="ISO format YYYY-MM-DD")
    vendor_name: Optional[str] = None
    buyer_name: Optional[str] = None
    total_amount: Optional[str] = Field(None, description="numeric string, no currency symbol")
    currency: Optional[str] = Field(None, description="e.g. INR")
    gstin: Optional[str] = Field(None, description="GST identification number if present")


class FormFields(BaseModel):
    doc_type: Literal["form"] = "form"
    form_title: Optional[str] = None
    applicant_name: Optional[str] = None
    applicant_name_hindi: Optional[str] = None
    date_of_birth: Optional[str] = None
    address: Optional[str] = None
    phone_number: Optional[str] = None
    form_id: Optional[str] = None


class IDCardFields(BaseModel):
    doc_type: Literal["id_card"] = "id_card"
    id_number: Optional[str] = None
    holder_name: Optional[str] = None
    holder_name_hindi: Optional[str] = None
    date_of_birth: Optional[str] = None
    issuing_authority: Optional[str] = None
    valid_until: Optional[str] = None


SCHEMA_REGISTRY = {
    "invoice": InvoiceFields,
    "form": FormFields,
    "id_card": IDCardFields,
}


def schema_for(doc_type: str) -> type[BaseModel]:
    if doc_type not in SCHEMA_REGISTRY:
        raise ValueError(f"Unknown doc_type '{doc_type}'. Options: {list(SCHEMA_REGISTRY)}")
    return SCHEMA_REGISTRY[doc_type]


def prompt_for(doc_type: str) -> str:
    """Instruction prompt paired with each image during training/inference."""
    fields = list(schema_for(doc_type).model_fields.keys())
    return (
        f"This is a scanned {doc_type.replace('_', ' ')}, possibly containing a mix of "
        f"English and Hindi text. Extract the following fields and return ONLY a JSON "
        f"object with these exact keys (use null for fields not present): {fields}."
    )
