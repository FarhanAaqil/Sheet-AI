# =============================================================================
# SheetSense AI — Dedicated Google Sheets Writer Module
# =============================================================================
# THIS IS THE ONLY MODULE PERMITTED TO CALL GSPREAD WRITE METHODS.
#
# Architectural Guardrail:
# - sheets_writer.py MUST ONLY be imported and called by the confirm handler
#   (POST /actions/{action_id}/confirm in main.py).
# - It is strictly forbidden for agent.py, tool definitions, or any query path
#   to import or call this module directly.
# =============================================================================

import logging
from typing import Any, Dict, Optional
import gspread

logger = logging.getLogger(__name__)


class SheetsWriterError(Exception):
    """Base exception for Google Sheets write execution errors."""
    pass


def execute_update_cell(
    spreadsheet: gspread.Spreadsheet,
    sheet_name: str,
    id_column: str,
    id_value: Any,
    update_column: str,
    new_value: Any,
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
            raise SheetsWriterError(f"Update column '{update_column}' not found in sheet '{sheet_name}'.")

        id_col_idx = headers.index(id_column) + 1
        upd_col_idx = headers.index(update_column) + 1

        cell = ws.find(str(id_value), in_column=id_col_idx)
        if not cell:
            raise SheetsWriterError(
                f"Row with {id_column}='{id_value}' not found in sheet '{sheet_name}'."
            )

        row_idx = cell.row
        # Write to Google Sheets
        ws.update_cell(row_idx, upd_col_idx, new_value)
        logger.info(
            f"Successfully updated cell at ({row_idx}, {upd_col_idx}) "
            f"in '{sheet_name}': {update_column} = {new_value}"
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
    except Exception as e:
        logger.error(f"Sheets update failed: {e}", exc_info=True)
        if isinstance(e, SheetsWriterError):
            raise e
        raise SheetsWriterError(f"Sheets update execution failed: {str(e)}") from e


def execute_delete_row(
    spreadsheet: gspread.Spreadsheet,
    sheet_name: str,
    id_column: str,
    id_value: Any,
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

        id_col_idx = headers.index(id_column) + 1
        cell = ws.find(str(id_value), in_column=id_col_idx)
        if not cell:
            raise SheetsWriterError(
                f"Row with {id_column}='{id_value}' not found in sheet '{sheet_name}'."
            )

        row_idx = cell.row
        # Delete row from Google Sheets
        ws.delete_rows(row_idx)
        logger.info(f"Successfully deleted row {row_idx} from '{sheet_name}' where {id_column}='{id_value}'.")

        return {
            "success": True,
            "sheet_name": sheet_name,
            "row_index": row_idx,
            "id_column": id_column,
            "id_value": str(id_value),
            "message": f"Deleted row where {id_column}='{id_value}' from '{sheet_name}'.",
        }
    except Exception as e:
        logger.error(f"Sheets delete failed: {e}", exc_info=True)
        if isinstance(e, SheetsWriterError):
            raise e
        raise SheetsWriterError(f"Sheets delete execution failed: {str(e)}") from e


def execute_action(
    spreadsheet: gspread.Spreadsheet,
    tool_name: str,
    target: Dict[str, Any],
    proposed_change: Dict[str, Any],
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
        )

    elif tool_name == "delete_row":
        return execute_delete_row(
            spreadsheet=spreadsheet,
            sheet_name=sheet_name,
            id_column=id_column,
            id_value=id_value,
        )

    else:
        raise SheetsWriterError(f"Unsupported destructive tool '{tool_name}'.")
