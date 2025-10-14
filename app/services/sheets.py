import base64
import json
from typing import Iterable, List

import gspread
from google.oauth2.service_account import Credentials

from app.core.config import settings

_CLIENT = None


def _client():
    """Lazily instantiate a Sheets client when credentials are configured."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    if not settings.GOOGLE_SHEETS_CREDENTIALS_JSON:
        return None
    raw = base64.b64decode(settings.GOOGLE_SHEETS_CREDENTIALS_JSON).decode("utf-8")
    creds_info = json.loads(raw)
    creds = Credentials.from_service_account_info(
        creds_info,
        scopes=["https://www.googleapis.com/auth/spreadsheets"],
    )
    _CLIENT = gspread.authorize(creds)
    return _CLIENT


def append_rows(values: Iterable[List[str | float | int]]) -> bool:
    """
    Append a batch of rows to the spreadsheet. Returns False if config is missing.
    """
    gc = _client()
    spreadsheet_id = settings.GOOGLE_SHEETS_SPREADSHEET_ID
    if gc is None or not spreadsheet_id:
        return False
    sh = gc.open_by_key(spreadsheet_id)
    worksheet_name = settings.GOOGLE_SHEETS_WORKSHEET_NAME
    ws = sh.worksheet(worksheet_name) if worksheet_name else sh.sheet1
    ws.append_rows(values, value_input_option="USER_ENTERED")
    return True
