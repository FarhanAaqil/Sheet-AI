# =============================================================================
# SheetSense AI — Static Analysis Unit Test
# =============================================================================

from scripts.check_write_isolation import check_write_isolation


def test_write_isolation_static_scan():
    """Ensure static analysis finds 0 unauthorized gspread write calls or writer imports."""
    violations = check_write_isolation()
    assert len(violations) == 0, f"Write isolation violations found:\n" + "\n".join(violations)
