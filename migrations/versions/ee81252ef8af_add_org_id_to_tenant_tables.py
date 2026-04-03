"""add_org_id_to_tenant_tables

Revision ID: ee81252ef8af
Revises: c3af57774955
Create Date: 2026-04-03 21:29:37.455193

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ee81252ef8af'
down_revision: Union[str, Sequence[str], None] = 'c3af57774955'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.

    Strategy:
    1. Add org_id as NULLABLE to all 15 tables
    2. Backfill all rows with org_id=1 (the default organization)
    3. Alter column to NOT NULL (using batch for SQLite)
    4. Create indexes
    5. Create foreign keys
    6. Fix account_settings unique constraint
    """
    # === STEP 1: Add org_id as NULLABLE ===
    op.add_column('account_settings', sa.Column('org_id', sa.Integer(), nullable=True))
    op.add_column('activities', sa.Column('org_id', sa.Integer(), nullable=True))
    op.add_column('alerts', sa.Column('org_id', sa.Integer(), nullable=True))
    op.add_column('audit_logs', sa.Column('org_id', sa.Integer(), nullable=True))
    op.add_column('firewall_ai_patterns', sa.Column('org_id', sa.Integer(), nullable=True))
    op.add_column('geo_rules', sa.Column('org_id', sa.Integer(), nullable=True))
    op.add_column('ip_blacklist', sa.Column('org_id', sa.Integer(), nullable=True))
    op.add_column('llm_endpoints', sa.Column('org_id', sa.Integer(), nullable=True))
    op.add_column('metrics', sa.Column('org_id', sa.Integer(), nullable=True))
    op.add_column('rate_limit_config', sa.Column('org_id', sa.Integer(), nullable=True))
    op.add_column('security_events', sa.Column('org_id', sa.Integer(), nullable=True))
    op.add_column('security_rules', sa.Column('org_id', sa.Integer(), nullable=True))
    op.add_column('threats', sa.Column('org_id', sa.Integer(), nullable=True))
    op.add_column('traffic_logs', sa.Column('org_id', sa.Integer(), nullable=True))
    op.add_column('users', sa.Column('org_id', sa.Integer(), nullable=True))

    # === STEP 2: Backfill all rows to org_id=1 ===
    op.execute("UPDATE account_settings SET org_id = 1 WHERE org_id IS NULL")
    op.execute("UPDATE activities SET org_id = 1 WHERE org_id IS NULL")
    op.execute("UPDATE alerts SET org_id = 1 WHERE org_id IS NULL")
    op.execute("UPDATE audit_logs SET org_id = 1 WHERE org_id IS NULL")
    op.execute("UPDATE firewall_ai_patterns SET org_id = 1 WHERE org_id IS NULL")
    op.execute("UPDATE geo_rules SET org_id = 1 WHERE org_id IS NULL")
    op.execute("UPDATE ip_blacklist SET org_id = 1 WHERE org_id IS NULL")
    op.execute("UPDATE llm_endpoints SET org_id = 1 WHERE org_id IS NULL")
    op.execute("UPDATE metrics SET org_id = 1 WHERE org_id IS NULL")
    op.execute("UPDATE rate_limit_config SET org_id = 1 WHERE org_id IS NULL")
    op.execute("UPDATE security_events SET org_id = 1 WHERE org_id IS NULL")
    op.execute("UPDATE security_rules SET org_id = 1 WHERE org_id IS NULL")
    op.execute("UPDATE threats SET org_id = 1 WHERE org_id IS NULL")
    op.execute("UPDATE traffic_logs SET org_id = 1 WHERE org_id IS NULL")
    op.execute("UPDATE users SET org_id = 1 WHERE org_id IS NULL")

    # === STEP 3: Alter columns to NOT NULL (use batch for SQLite) ===
    with op.batch_alter_table('account_settings') as batch_op:
        batch_op.alter_column('org_id', nullable=False, existing_type=sa.Integer())
    with op.batch_alter_table('activities') as batch_op:
        batch_op.alter_column('org_id', nullable=False, existing_type=sa.Integer())
    with op.batch_alter_table('alerts') as batch_op:
        batch_op.alter_column('org_id', nullable=False, existing_type=sa.Integer())
    with op.batch_alter_table('audit_logs') as batch_op:
        batch_op.alter_column('org_id', nullable=False, existing_type=sa.Integer())
    with op.batch_alter_table('firewall_ai_patterns') as batch_op:
        batch_op.alter_column('org_id', nullable=False, existing_type=sa.Integer())
    with op.batch_alter_table('geo_rules') as batch_op:
        batch_op.alter_column('org_id', nullable=False, existing_type=sa.Integer())
    with op.batch_alter_table('ip_blacklist') as batch_op:
        batch_op.alter_column('org_id', nullable=False, existing_type=sa.Integer())
    with op.batch_alter_table('llm_endpoints') as batch_op:
        batch_op.alter_column('org_id', nullable=False, existing_type=sa.Integer())
    with op.batch_alter_table('metrics') as batch_op:
        batch_op.alter_column('org_id', nullable=False, existing_type=sa.Integer())
    with op.batch_alter_table('rate_limit_config') as batch_op:
        batch_op.alter_column('org_id', nullable=False, existing_type=sa.Integer())
    with op.batch_alter_table('security_events') as batch_op:
        batch_op.alter_column('org_id', nullable=False, existing_type=sa.Integer())
    with op.batch_alter_table('security_rules') as batch_op:
        batch_op.alter_column('org_id', nullable=False, existing_type=sa.Integer())
    with op.batch_alter_table('threats') as batch_op:
        batch_op.alter_column('org_id', nullable=False, existing_type=sa.Integer())
    with op.batch_alter_table('traffic_logs') as batch_op:
        batch_op.alter_column('org_id', nullable=False, existing_type=sa.Integer())
    with op.batch_alter_table('users') as batch_op:
        batch_op.alter_column('org_id', nullable=False, existing_type=sa.Integer())

    # === STEP 4: Handle account_settings unique constraint (also batch for SQLite) ===
    with op.batch_alter_table('account_settings') as batch_op:
        batch_op.drop_index('ix_account_settings_key')
        batch_op.create_unique_constraint('uq_account_settings_org_key', ['org_id', 'key'])

    # === STEP 5: Create indexes for org_id ===
    op.create_index(op.f('ix_activities_org_id'), 'activities', ['org_id'], unique=False)
    op.create_index(op.f('ix_alerts_org_id'), 'alerts', ['org_id'], unique=False)
    op.create_index(op.f('ix_audit_logs_org_id'), 'audit_logs', ['org_id'], unique=False)
    op.create_index(op.f('ix_firewall_ai_patterns_org_id'), 'firewall_ai_patterns', ['org_id'], unique=False)
    op.create_index(op.f('ix_geo_rules_org_id'), 'geo_rules', ['org_id'], unique=False)
    op.create_index(op.f('ix_ip_blacklist_org_id'), 'ip_blacklist', ['org_id'], unique=False)
    op.create_index(op.f('ix_llm_endpoints_org_id'), 'llm_endpoints', ['org_id'], unique=False)
    op.create_index(op.f('ix_metrics_org_id'), 'metrics', ['org_id'], unique=False)
    op.create_index(op.f('ix_rate_limit_config_org_id'), 'rate_limit_config', ['org_id'], unique=False)
    op.create_index(op.f('ix_security_events_org_id'), 'security_events', ['org_id'], unique=False)
    op.create_index(op.f('ix_security_rules_org_id'), 'security_rules', ['org_id'], unique=False)
    op.create_index(op.f('ix_threats_org_id'), 'threats', ['org_id'], unique=False)
    op.create_index(op.f('ix_traffic_logs_org_id'), 'traffic_logs', ['org_id'], unique=False)
    op.create_index(op.f('ix_users_org_id'), 'users', ['org_id'], unique=False)
    op.create_index(op.f('ix_account_settings_org_id'), 'account_settings', ['org_id'], unique=False)

    # === STEP 6: Create foreign keys ===
    op.create_foreign_key('fk_account_settings_org_id', 'account_settings', 'organizations', ['org_id'], ['id'], ondelete='CASCADE')
    op.create_foreign_key('fk_activities_org_id', 'activities', 'organizations', ['org_id'], ['id'], ondelete='CASCADE')
    op.create_foreign_key('fk_alerts_org_id', 'alerts', 'organizations', ['org_id'], ['id'], ondelete='CASCADE')
    op.create_foreign_key('fk_audit_logs_org_id', 'audit_logs', 'organizations', ['org_id'], ['id'], ondelete='CASCADE')
    op.create_foreign_key('fk_firewall_ai_patterns_org_id', 'firewall_ai_patterns', 'organizations', ['org_id'], ['id'], ondelete='CASCADE')
    op.create_foreign_key('fk_geo_rules_org_id', 'geo_rules', 'organizations', ['org_id'], ['id'], ondelete='CASCADE')
    op.create_foreign_key('fk_ip_blacklist_org_id', 'ip_blacklist', 'organizations', ['org_id'], ['id'], ondelete='CASCADE')
    op.create_foreign_key('fk_llm_endpoints_org_id', 'llm_endpoints', 'organizations', ['org_id'], ['id'], ondelete='CASCADE')
    op.create_foreign_key('fk_metrics_org_id', 'metrics', 'organizations', ['org_id'], ['id'], ondelete='CASCADE')
    op.create_foreign_key('fk_rate_limit_config_org_id', 'rate_limit_config', 'organizations', ['org_id'], ['id'], ondelete='CASCADE')
    op.create_foreign_key('fk_security_events_org_id', 'security_events', 'organizations', ['org_id'], ['id'], ondelete='CASCADE')
    op.create_foreign_key('fk_security_rules_org_id', 'security_rules', 'organizations', ['org_id'], ['id'], ondelete='CASCADE')
    op.create_foreign_key('fk_threats_org_id', 'threats', 'organizations', ['org_id'], ['id'], ondelete='CASCADE')
    op.create_foreign_key('fk_traffic_logs_org_id', 'traffic_logs', 'organizations', ['org_id'], ['id'], ondelete='CASCADE')
    op.create_foreign_key('fk_users_org_id', 'users', 'organizations', ['org_id'], ['id'], ondelete='CASCADE')


def downgrade() -> None:
    """Downgrade schema."""
    # Drop foreign keys
    op.drop_constraint('fk_users_org_id', 'users', type_='foreignkey')
    op.drop_constraint('fk_traffic_logs_org_id', 'traffic_logs', type_='foreignkey')
    op.drop_constraint('fk_threats_org_id', 'threats', type_='foreignkey')
    op.drop_constraint('fk_security_rules_org_id', 'security_rules', type_='foreignkey')
    op.drop_constraint('fk_security_events_org_id', 'security_events', type_='foreignkey')
    op.drop_constraint('fk_rate_limit_config_org_id', 'rate_limit_config', type_='foreignkey')
    op.drop_constraint('fk_metrics_org_id', 'metrics', type_='foreignkey')
    op.drop_constraint('fk_llm_endpoints_org_id', 'llm_endpoints', type_='foreignkey')
    op.drop_constraint('fk_ip_blacklist_org_id', 'ip_blacklist', type_='foreignkey')
    op.drop_constraint('fk_geo_rules_org_id', 'geo_rules', type_='foreignkey')
    op.drop_constraint('fk_firewall_ai_patterns_org_id', 'firewall_ai_patterns', type_='foreignkey')
    op.drop_constraint('fk_audit_logs_org_id', 'audit_logs', type_='foreignkey')
    op.drop_constraint('fk_alerts_org_id', 'alerts', type_='foreignkey')
    op.drop_constraint('fk_activities_org_id', 'activities', type_='foreignkey')
    op.drop_constraint('fk_account_settings_org_id', 'account_settings', type_='foreignkey')

    # Drop indexes for org_id
    op.drop_index(op.f('ix_account_settings_org_id'), table_name='account_settings')
    op.drop_index(op.f('ix_activities_org_id'), table_name='activities')
    op.drop_index(op.f('ix_alerts_org_id'), table_name='alerts')
    op.drop_index(op.f('ix_audit_logs_org_id'), table_name='audit_logs')
    op.drop_index(op.f('ix_firewall_ai_patterns_org_id'), table_name='firewall_ai_patterns')
    op.drop_index(op.f('ix_geo_rules_org_id'), table_name='geo_rules')
    op.drop_index(op.f('ix_ip_blacklist_org_id'), table_name='ip_blacklist')
    op.drop_index(op.f('ix_llm_endpoints_org_id'), table_name='llm_endpoints')
    op.drop_index(op.f('ix_metrics_org_id'), table_name='metrics')
    op.drop_index(op.f('ix_rate_limit_config_org_id'), table_name='rate_limit_config')
    op.drop_index(op.f('ix_security_events_org_id'), table_name='security_events')
    op.drop_index(op.f('ix_security_rules_org_id'), table_name='security_rules')
    op.drop_index(op.f('ix_threats_org_id'), table_name='threats')
    op.drop_index(op.f('ix_traffic_logs_org_id'), table_name='traffic_logs')
    op.drop_index(op.f('ix_users_org_id'), table_name='users')

    # Restore account_settings unique constraint
    with op.batch_alter_table('account_settings') as batch_op:
        batch_op.drop_constraint('uq_account_settings_org_key', type_='unique')
        batch_op.create_index('ix_account_settings_key', ['key'], unique=True)

    # Drop columns (use batch for SQLite)
    with op.batch_alter_table('account_settings') as batch_op:
        batch_op.drop_column('org_id')
    with op.batch_alter_table('activities') as batch_op:
        batch_op.drop_column('org_id')
    with op.batch_alter_table('alerts') as batch_op:
        batch_op.drop_column('org_id')
    with op.batch_alter_table('audit_logs') as batch_op:
        batch_op.drop_column('org_id')
    with op.batch_alter_table('firewall_ai_patterns') as batch_op:
        batch_op.drop_column('org_id')
    with op.batch_alter_table('geo_rules') as batch_op:
        batch_op.drop_column('org_id')
    with op.batch_alter_table('ip_blacklist') as batch_op:
        batch_op.drop_column('org_id')
    with op.batch_alter_table('llm_endpoints') as batch_op:
        batch_op.drop_column('org_id')
    with op.batch_alter_table('metrics') as batch_op:
        batch_op.drop_column('org_id')
    with op.batch_alter_table('rate_limit_config') as batch_op:
        batch_op.drop_column('org_id')
    with op.batch_alter_table('security_events') as batch_op:
        batch_op.drop_column('org_id')
    with op.batch_alter_table('security_rules') as batch_op:
        batch_op.drop_column('org_id')
    with op.batch_alter_table('threats') as batch_op:
        batch_op.drop_column('org_id')
    with op.batch_alter_table('traffic_logs') as batch_op:
        batch_op.drop_column('org_id')
    with op.batch_alter_table('users') as batch_op:
        batch_op.drop_column('org_id')
