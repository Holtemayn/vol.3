
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from datetime import date
from app.services.sheets import append_rows

router = APIRouter()

class ReconcileRow(BaseModel):
    date: date
    revenue_pred: float
    hours_recommended: float
    planday_hours: Optional[float] = None
    hours_diff: Optional[float] = None

class ReconcileRequest(BaseModel):
    rows: List[ReconcileRow]

@router.post("")
def reconcile(req: ReconcileRequest):
    values = [[
        r.date.isoformat(),
        r.revenue_pred,
        r.hours_recommended,
        r.planday_hours if r.planday_hours is not None else "",
        r.hours_diff if r.hours_diff is not None else ""
    ] for r in req.rows]
    ok = append_rows(values)
    return {"status": "ok" if ok else "skipped", "rows": len(values)}
