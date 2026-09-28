"""Initial attendance schema with pgvector.

Revision ID: 0001_initial
"""
from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("""
        CREATE TABLE people (
            id SERIAL PRIMARY KEY,
            employee_code VARCHAR(100) NOT NULL UNIQUE,
            name VARCHAR(255) NOT NULL,
            email VARCHAR(320),
            active BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("""
        CREATE TABLE doors (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            location VARCHAR(255),
            active BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("""
        CREATE TABLE face_embeddings (
            id SERIAL PRIMARY KEY,
            person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
            embedding VECTOR(128) NOT NULL,
            model_name VARCHAR(100) NOT NULL,
            model_version VARCHAR(100) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX ix_face_embeddings_embedding_cosine ON face_embeddings USING hnsw (embedding vector_cosine_ops)")
    op.execute("""
        CREATE TABLE attendance (
            id SERIAL PRIMARY KEY,
            person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
            door_id INTEGER REFERENCES doors(id) ON DELETE SET NULL,
            event_type VARCHAR(50) NOT NULL DEFAULT 'check_in',
            event_at TIMESTAMPTZ NOT NULL,
            attendance_date DATE NOT NULL,
            similarity DOUBLE PRECISION NOT NULL,
            liveness_score DOUBLE PRECISION NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT uq_attendance_person_day_event UNIQUE (person_id, attendance_date, event_type)
        )
    """)
    op.execute("CREATE INDEX ix_attendance_attendance_date ON attendance(attendance_date)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS attendance")
    op.execute("DROP TABLE IF EXISTS face_embeddings")
    op.execute("DROP TABLE IF EXISTS doors")
    op.execute("DROP TABLE IF EXISTS people")
