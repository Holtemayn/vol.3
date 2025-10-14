
from fastapi import APIRouter
from datetime import date
from app.services.planday import get_planday_hours_for_dates

router = APIRouter()

@router.get("/{dt}")
def get_planday(dt: date):
    res = get_planday_hours_for_dates([dt])
    return {"date": dt.isoformat(), "hours": res.get(dt)}
