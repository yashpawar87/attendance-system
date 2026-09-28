"""Add employee profile photos for the dashboard.

Revision ID: 0002_profile_photos
Revises: 0001_initial
"""
from alembic import op
import sqlalchemy as sa


revision = "0002_profile_photos"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("people", sa.Column("profile_photo", sa.LargeBinary(), nullable=True))
    op.add_column("people", sa.Column("profile_photo_content_type", sa.String(length=100), nullable=True))


def downgrade() -> None:
    op.drop_column("people", "profile_photo_content_type")
    op.drop_column("people", "profile_photo")
