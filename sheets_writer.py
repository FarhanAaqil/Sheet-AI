# SheetSense AI — Dedicated Google Sheets Writer Module
# THIS IS THE ONLY MODULE PERMITTED TO CALL GSPREAD WRITE METHODS.
#
# Architectural Guardrail:
# - sheets_writer.py MUST ONLY be imported and called by the confirm handler
#   (POST /actions/{action_id}/confirm in main.py).
# - It is strictly forbidden for agent.py, tool definitions, or any query path
#   to import or call this module directly.

import hashlib
import json
import logging
from typing import Any, Dict, Optional

import gspread

logger = logging.getLogger(__name__)


class SheetsWriterError(Exception):
    """Base exception for Google Sheets write execution errors."""

    pass


class SheetsStateConflictError(SheetsWriterError):
    """The live row no longer matches the state approved by the user."""


def row_fingerprint(headers, row_values) -> str:
    normalized = {
        str(header): str(row_values[index]) if index < len(row_values) else ""
        for index, header in enumerate(headers)
    }
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolve_unique_row(ws, id_column: str, id_value: Any, headers) -> int:
    id_col_idx = headers.index(id_column) + 1
    matches = ws.findall(str(id_value), in_column=id_col_idx)
    if isinstance(matches, list):
        if len(matches) != 1:
            raise SheetsStateConflictError("The target identifier is missing or no longer unique.")
        return matches[0].row
    cell = ws.find(str(id_value), in_column=id_col_idx)
    if not cell:
        raise SheetsStateConflictError("The target row no longer exists.")
    return cell.row


def _verify_expected_state(
    ws, headers, row_idx: int, expected_state: Optional[Dict[str, Any]]
) -> None:
    if not expected_state:
        return
    live_values = ws.row_values(row_idx)
    live_fingerprint = row_fingerprint(headers, live_values)
    if live_fingerprint != expected_state.get("row_fingerprint"):
        raise SheetsStateConflictError("The target row changed after the action was staged.")


def execute_update_cell(
    spreadsheet: gspread.Spreadsheet,
    sheet_name: str,
    id_column: str,
    id_value: Any,
    update_column: str,
    new_value: Any,
    expected_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Locate a row by id_column == id_value and update the specified cell.

    Args:
        spreadsheet: Authenticated gspread Spreadsheet instance.
        sheet_name: Name of the worksheet.
        id_column: Header name to search for the identifier.
        id_value: Target identifier value.
        update_column: Header name of the column to update.
        new_value: New value to set.

    Returns:
        Dict with execution status and change summary.
    """
    try:
        ws = spreadsheet.worksheet(sheet_name)
        headers = ws.row_values(1)

        if id_column not in headers:
            raise SheetsWriterError(f"ID column '{id_column}' not found in sheet '{sheet_name}'.")
        if update_column not in headers:
            raise SheetsWriterError(
                f"Update column '{update_column}' not found in sheet '{sheet_name}'."
            )

        upd_col_idx = headers.index(update_column) + 1
        row_idx = _resolve_unique_row(ws, id_column, id_value, headers)
        _verify_expected_state(ws, headers, row_idx, expected_state)
        # Write to Google Sheets
        ws.update_cell(row_idx, upd_col_idx, new_value)
        logger.info(
            "Google Sheets cell updated: sheet=%s row=%s column=%s",
            sheet_name,
            row_idx,
            upd_col_idx,
        )

        return {
            "success": True,
            "sheet_name": sheet_name,
            "row_index": row_idx,
            "id_column": id_column,
            "id_value": str(id_value),
            "update_column": update_column,
            "new_value": new_value,
            "message": f"Updated '{update_column}' to '{new_value}' where {id_column}='{id_value}'.",
        }
    except Exception as exc:
        logger.error("Sheets update failed: exception_type=%s", type(exc).__name__)
        if isinstance(exc, SheetsWriterError):
            raise exc
        raise SheetsWriterError("Sheets update execution failed.") from exc


def execute_delete_row(
    spreadsheet: gspread.Spreadsheet,
    sheet_name: str,
    id_column: str,
    id_value: Any,
    expected_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Locate a row by id_column == id_value and permanently delete it.

    Args:
        spreadsheet: Authenticated gspread Spreadsheet instance.
        sheet_name: Name of the worksheet.
        id_column: Header name used to locate the row.
        id_value: Target identifier value.

    Returns:
        Dict with execution status and deletion summary.
    """
    try:
        ws = spreadsheet.worksheet(sheet_name)
        headers = ws.row_values(1)

        if id_column not in headers:
            raise SheetsWriterError(f"ID column '{id_column}' not found in sheet '{sheet_name}'.")

        row_idx = _resolve_unique_row(ws, id_column, id_value, headers)
        _verify_expected_state(ws, headers, row_idx, expected_state)
        # Delete row from Google Sheets
        ws.delete_rows(row_idx)
        logger.info("Google Sheets row deleted: sheet=%s row=%s", sheet_name, row_idx)

        return {
            "success": True,
            "sheet_name": sheet_name,
            "row_index": row_idx,
            "id_column": id_column,
            "id_value": str(id_value),
            "message": f"Deleted row where {id_column}='{id_value}' from '{sheet_name}'.",
        }
    except Exception as exc:
        logger.error("Sheets delete failed: exception_type=%s", type(exc).__name__)
        if isinstance(exc, SheetsWriterError):
            raise exc
        raise SheetsWriterError("Sheets delete execution failed.") from exc


def execute_action(
    spreadsheet: gspread.Spreadsheet,
    tool_name: str,
    target: Dict[str, Any],
    proposed_change: Dict[str, Any],
    expected_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Dispatch and execute a confirmed destructive action against Google Sheets.

    Args:
        spreadsheet: Authenticated gspread Spreadsheet instance.
        tool_name: 'update_cell' or 'delete_row'.
        target: Dict with target parameters (e.g. sheet_name, id_column, id_value).
        proposed_change: Dict with change parameters (e.g. update_column, new_value, or action='delete').
    """
    sheet_name = target.get("sheet_name")
    id_column = target.get("id_column")
    id_value = target.get("id_value")

    if not sheet_name or not id_column or id_value is None:
        raise SheetsWriterError(
            f"Invalid target specification: sheet_name, id_column, and id_value are required. Got: {target}"
        )

    if tool_name == "update_cell":
        update_column = proposed_change.get("update_column")
        new_value = proposed_change.get("new_value")
        if not update_column:
            raise SheetsWriterError("Missing 'update_column' in proposed_change.")
        return execute_update_cell(
            spreadsheet=spreadsheet,
            sheet_name=sheet_name,
            id_column=id_column,
            id_value=id_value,
            update_column=update_column,
            new_value=new_value,
            expected_state=expected_state,
        )

    elif tool_name == "delete_row":
        return execute_delete_row(
            spreadsheet=spreadsheet,
            sheet_name=sheet_name,
            id_column=id_column,
            id_value=id_value,
            expected_state=expected_state,
        )

    else:
        raise SheetsWriterError(f"Unsupported destructive tool '{tool_name}'.")
