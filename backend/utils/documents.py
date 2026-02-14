def classify_document(filename: str, content_type: str, text: str) -> str:
    t = (text or "").lower()
    name = (filename or "").lower()

    if "invoice" in name or "bill" in name or "gst" in t or "total" in t:
        return "itemized_invoice"
    if "receipt" in name or "payment" in name or "paid on" in t or "receipt" in t:
        return "payment_receipt"
    if "fir" in name or "first information report" in t or "police station" in t:
        return "fir"
    if "discharge" in name or "admission date" in t or "discharge date" in t:
        return "discharge_summary"
    if "id" in name or "aadhaar" in name or "pan" in name or "passport" in t:
        return "id_proof"
    return "unknown"