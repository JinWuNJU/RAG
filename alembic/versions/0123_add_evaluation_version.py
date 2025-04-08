from alembic import op
import sqlalchemy as sa

def upgrade():
    op.add_column('evaluation_tasks', sa.Column('version', sa.Integer(), server_default='1'))
    op.add_column('evaluation_tasks', sa.Column('previous_version', sa.UUID(), nullable=True))

def downgrade():
    op.drop_column('evaluation_tasks', 'version')
    op.drop_column('evaluation_tasks', 'previous_version')