"""add knowledge_base_count to users table

Revision ID: a6fe8bbde443
Revises: cdb8ea3495d7
Create Date: 2025-05-20 19:31:58.680306

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a6fe8bbde443'
down_revision: Union[str, None] = 'cdb8ea3495d7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.add_column('users', sa.Column('knowledge_base_count', sa.Integer(), nullable=False, server_default='0'))

def downgrade():
    op.drop_column('users', 'knowledge_base_count')