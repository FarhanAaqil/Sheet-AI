"""Phase 3 production schema with ownership and state constraints."""

import sqlalchemy as sa
from alembic import op

revision = "20261024_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "tool_calls",
        sa.Column("call_id", sa.BigInteger(), primary_key=True),
        sa.Column("session_id", sa.Text()),
        sa.Column("tool_name", sa.Text(), nullable=False),
        sa.Column("input_json", sa.Text()),
        sa.Column("output_json", sa.Text()),
        sa.Column("success", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("latency_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("request_id", sa.Text()),
        sa.Column("trace_id", sa.Text()),
    )
    op.create_table(
        "pending_actions",
        sa.Column("action_id", sa.Text(), primary_key=True),
        sa.Column("session_id", sa.Text()),
        sa.Column("principal_id", sa.Text(), nullable=False),
        sa.Column("connection_id", sa.Text()),
        sa.Column("tool_name", sa.Text(), nullable=False),
        sa.Column("target_json", sa.Text(), nullable=False),
        sa.Column("proposed_change_json", sa.Text(), nullable=False),
        sa.Column("expected_state_json", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("claimed_at", sa.DateTime(timezone=True)),
        sa.Column("executed_at", sa.DateTime(timezone=True)),
        sa.Column("failure_code", sa.Text()),
        sa.Column("idempotency_key", sa.Text()),
        sa.Column("result_json", sa.Text()),
        sa.CheckConstraint(
            "status IN ('pending','executing','confirmed','failed','rejected','expired')",
            name="ck_pending_actions_status",
        ),
    )
    op.create_index(
        "uq_action_owner_idempotency",
        "pending_actions",
        ["principal_id", "idempotency_key"],
        unique=True,
        postgresql_where=sa.text("idempotency_key IS NOT NULL"),
    )
    op.create_index(
        "ix_pending_owner_status",
        "pending_actions",
        ["principal_id", "status", "expires_at"],
    )
    op.create_table(
        "spreadsheet_connections",
        sa.Column("connection_id", sa.Text(), primary_key=True),
        sa.Column("principal_id", sa.Text(), nullable=False),
        sa.Column("spreadsheet_id", sa.Text(), nullable=False),
        sa.Column("display_name", sa.Text(), nullable=False),
        sa.Column("permission_level", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_sync_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("principal_id", "spreadsheet_id"),
        sa.CheckConstraint(
            "permission_level IN ('viewer','editor')",
            name="ck_connections_permission",
        ),
    )
    op.create_table(
        "eval_runs",
        sa.Column("eval_id", sa.Text(), primary_key=True),
        sa.Column("total_queries", sa.Integer(), nullable=False),
        sa.Column("tool_selection_accuracy", sa.Float(), nullable=False),
        sa.Column("answer_correctness_rate", sa.Float(), nullable=False),
        sa.Column("guardrail_compliance_rate", sa.Float(), nullable=False),
        sa.Column("run_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("mode", sa.Text(), nullable=False),
        sa.Column("model_name", sa.Text()),
        sa.Column("model_version", sa.Text()),
        sa.Column("dataset_version", sa.Text()),
        sa.Column("metrics_json", sa.Text()),
        sa.Column("baseline_delta_json", sa.Text()),
    )
    op.create_table(
        "eval_results",
        sa.Column("run_id", sa.Text(), sa.ForeignKey("eval_runs.eval_id"), primary_key=True),
        sa.Column("case_id", sa.Text(), primary_key=True),
        sa.Column("category", sa.Text(), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("result_json", sa.Text(), nullable=False),
    )
    op.create_table(
        "audit_events",
        sa.Column("event_id", sa.BigInteger(), primary_key=True),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("principal_id", sa.Text()),
        sa.Column("request_id", sa.Text()),
        sa.Column("trace_id", sa.Text()),
        sa.Column("resource_id", sa.Text()),
        sa.Column("outcome", sa.Text(), nullable=False),
        sa.Column("detail_json", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade():
    for table in (
        "audit_events",
        "eval_results",
        "eval_runs",
        "spreadsheet_connections",
        "pending_actions",
        "tool_calls",
    ):
        op.drop_table(table)
