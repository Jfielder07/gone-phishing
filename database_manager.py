"""
DatabaseManager
---------------
Saves uploaded emails and their analysis results to PostgreSQL (hosted on Neon),
and reads results back.

The connection string comes from the DATABASE_URL environment variable, so the
password never appears in the code or on GitHub. Locally it lives in a .env file;
on Render it's set in the service's Environment settings.

The tables live in the schema named by DATABASE_SCHEMA (default "public"). If
the tables already exist there, connect() checks their columns match what this
code needs and explains any mismatch, instead of failing later on the first save.
If they don't exist yet, connect() creates them.

Usage:
    db = DatabaseManager()
    db.connect()
    email_id = db.save_email(email)            # also sets email.email_id
    result_id = db.save_result(result, email_id)  # also sets result.result_id
    db.get_result(result_id)                   # -> AnalysisResult, or None
"""
import os

import psycopg
from psycopg import sql

from analysis_result import AnalysisResult

DEFAULT_SCHEMA = "public"

TEXT_TYPES = ("text", "character varying", "character")
INT_TYPES = ("smallint", "integer", "bigint")
NUMBER_TYPES = INT_TYPES + ("numeric", "real", "double precision")

# Every column this code reads or writes, and the column types that work for it
REQUIRED_COLUMNS = {
    "emails": {"email_id": INT_TYPES, "filename": TEXT_TYPES, "sender": TEXT_TYPES,
               "subject": TEXT_TYPES, "body": TEXT_TYPES, "links": ("ARRAY",)},
    "analysis_results": {"result_id": INT_TYPES, "email_id": INT_TYPES, "classification": TEXT_TYPES,
                         "safety_score": NUMBER_TYPES, "explanation": TEXT_TYPES,
                         "explanation_status": TEXT_TYPES},
}
# The database must fill these in by itself (IDENTITY or SERIAL), because inserts don't supply them
ID_COLUMNS = {"emails": "email_id", "analysis_results": "result_id"}

SCHEMA_EXISTS_SQL = "SELECT 1 FROM pg_namespace WHERE nspname = %s"

COLUMNS_SQL = """
SELECT table_name, column_name, data_type, is_nullable, column_default, is_identity
FROM information_schema.columns
WHERE table_schema = %s AND table_name IN ('emails', 'analysis_results')
ORDER BY table_name, ordinal_position
"""

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS emails (
    email_id     BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    filename     TEXT        NOT NULL,
    sender       TEXT        NOT NULL DEFAULT '',
    subject      TEXT        NOT NULL DEFAULT '',
    body         TEXT        NOT NULL DEFAULT '',
    links        TEXT[]      NOT NULL DEFAULT '{}',
    uploaded_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS analysis_results (
    result_id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    email_id           BIGINT      NOT NULL REFERENCES emails (email_id) ON DELETE CASCADE,
    classification     TEXT        NOT NULL CHECK (classification IN ('safe', 'phishing')),
    safety_score       SMALLINT    NOT NULL CHECK (safety_score BETWEEN 0 AND 100),
    explanation        TEXT,
    explanation_status TEXT        NOT NULL
                       CHECK (explanation_status IN ('not_needed', 'available', 'unavailable')),
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS analysis_results_email_id_idx ON analysis_results (email_id);
"""

INSERT_EMAIL_SQL = """
INSERT INTO emails (filename, sender, subject, body, links)
VALUES (%s, %s, %s, %s, %s)
RETURNING email_id
"""

INSERT_RESULT_SQL = """
INSERT INTO analysis_results (email_id, classification, safety_score, explanation, explanation_status)
VALUES (%s, %s, %s, %s, %s)
RETURNING result_id
"""

SELECT_RESULT_SQL = """
SELECT result_id, classification, safety_score, explanation, explanation_status
FROM analysis_results
WHERE result_id = %s
"""


class DatabaseError(Exception):
    """Raised when the database can't be reached or a query fails.

    The rest of the program catches this one error type, so it never has to
    know about psycopg's own error classes.
    """


class DatabaseManager:
    def __init__(self, database_url=None, schema=None):
        """
        database_url: a PostgreSQL connection string. Leave it out to read the
                      DATABASE_URL environment variable (the normal case).
        schema:       which schema the tables are in. Leave it out to read
                      DATABASE_SCHEMA, or use "public".
        """
        self.database_url = database_url or os.environ.get("DATABASE_URL")
        self.schema = schema or os.environ.get("DATABASE_SCHEMA") or DEFAULT_SCHEMA
        self.connection = None

    def is_configured(self):
        """True if there's a connection string to use."""
        return bool(self.database_url)

    def connect(self):
        """
        Open a connection (if one isn't already open). If the tables don't exist in
        the schema yet, create them; if they do, check they're compatible.
        Safe to call as often as you like.
        """
        if self.connection is not None and not self.connection.closed:
            return self.connection

        connection = self._open()
        try:
            with connection.cursor() as cur:
                tables = self._describe(cur)
                if not tables:
                    cur.execute(CREATE_TABLES_SQL)  # created in self.schema (it's first on the search path)
            connection.commit()
        except psycopg.Error as error:
            connection.close()
            raise DatabaseError(f"Could not set up the tables: {error}") from error

        problems = find_table_problems(tables) if tables else []
        if problems:
            connection.close()
            raise DatabaseError(f"The tables in schema {self.schema!r} don't match what the app needs: "
                                + " | ".join(problems))
        self.connection = connection
        return self.connection

    def describe_tables(self):
        """
        Read-only look at the existing tables: {table: [column info dicts]}.
        Uses its own short-lived connection and never creates anything.
        """
        connection = self._open()
        try:
            with connection.cursor() as cur:
                return self._describe(cur)
        finally:
            connection.close()

    def save_email(self, email):
        """
        Insert an Email and return its new email_id.
        Also stores the ID on the object, so email.email_id is filled in afterwards.
        """
        row = self._run(INSERT_EMAIL_SQL, (
            email.filename,
            self._clean(email.sender),
            self._clean(email.subject),
            self._clean(email.body),
            [self._clean(link) for link in email.links],
        ))
        email.email_id = row[0]
        return email.email_id

    def save_result(self, result, email_id):
        """
        Insert an AnalysisResult linked to the email it belongs to, and return
        its new result_id. Also sets result.result_id.
        """
        row = self._run(INSERT_RESULT_SQL, (
            email_id,
            result.classification,
            result.safety_score,
            self._clean(result.explanation),
            result.explanation_status,
        ))
        result.result_id = row[0]
        return result.result_id

    def get_result(self, result_id):
        """Return the saved AnalysisResult with this ID, or None if there isn't one."""
        row = self._run(SELECT_RESULT_SQL, (result_id,), write=False)
        if row is None:
            return None
        result_id, classification, safety_score, explanation, explanation_status = row
        return AnalysisResult(
            classification,
            safety_score,
            explanation=explanation,
            explanation_status=explanation_status,
            result_id=result_id,
        )

    def close(self):
        """Close the connection, e.g. when the server shuts down."""
        if self.connection is not None and not self.connection.closed:
            self.connection.close()
        self.connection = None

    # ---- private helpers ----
    def _open(self):
        """Connect, check the schema exists, and make it the place unqualified table names refer to."""
        if not self.is_configured():
            raise DatabaseError("DATABASE_URL is not set, so there is no database to connect to.")
        try:
            connection = psycopg.connect(
                self.database_url,
                connect_timeout=10,      # give up instead of hanging forever
                prepare_threshold=None,  # avoids errors if Neon's pooled connection is used
            )
        except psycopg.Error as error:
            raise DatabaseError(f"Could not connect to the database: {error}") from error
        try:
            with connection.cursor() as cur:
                cur.execute(SCHEMA_EXISTS_SQL, (self.schema,))
                if cur.fetchone() is None:
                    raise DatabaseError(f"Schema {self.schema!r} doesn't exist in this database. "
                                        "Check DATABASE_SCHEMA.")
                # From now on, "emails" means "<schema>.emails" on this connection
                cur.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(self.schema)))
            connection.commit()
        except psycopg.Error as error:
            connection.close()
            raise DatabaseError(f"Could not use schema {self.schema!r}: {error}") from error
        except DatabaseError:
            connection.close()
            raise
        return connection

    def _describe(self, cur):
        cur.execute(COLUMNS_SQL, (self.schema,))
        names = [d.name for d in cur.description]
        tables = {}
        for row in cur.fetchall():
            info = dict(zip(names, row))
            tables.setdefault(info["table_name"], []).append(info)
        return tables

    def _run(self, query, params, write=True):
        """
        Run one query and return its first row.

        Neon closes idle connections after a few minutes, so a connection that
        worked earlier may be dead now. If that happens, reconnect and try once more.
        """
        for attempt in (1, 2):
            self.connect()
            try:
                with self.connection.cursor() as cur:
                    cur.execute(query, params)
                    row = cur.fetchone()
                if write:
                    self.connection.commit()
                else:
                    self.connection.rollback()  # end the read-only transaction
                return row
            except psycopg.OperationalError as error:
                # The connection dropped. Throw it away and retry once with a fresh one.
                self.close()
                if attempt == 2:
                    raise DatabaseError(f"Lost the database connection: {error}") from error
            except psycopg.Error as error:
                # A real problem with the query or data: undo it and report it.
                self.connection.rollback()
                raise DatabaseError(f"Database query failed: {error}") from error

    @staticmethod
    def _clean(text):
        """PostgreSQL text can't contain null characters, which a few odd emails do."""
        return text.replace("\x00", "") if isinstance(text, str) else text


def find_table_problems(tables):
    """
    Compare existing tables ({table: [column info]}, as from describe_tables())
    with what this code needs. Returns a list of plain-English problems; empty = compatible.
    """
    problems = []
    for table, needed in REQUIRED_COLUMNS.items():
        if table not in tables:
            problems.append(f"table {table} is missing")
            continue
        columns = {c["column_name"]: c for c in tables[table]}
        for name, ok_types in needed.items():
            if name not in columns:
                problems.append(f"{table} has no column {name}")
            elif columns[name]["data_type"] not in ok_types:
                problems.append(f"{table}.{name} is {columns[name]['data_type']}, "
                                f"expected {' or '.join(ok_types)}")
        id_col = columns.get(ID_COLUMNS[table])
        if id_col and not (id_col.get("column_default") or id_col.get("is_identity") == "YES"):
            problems.append(f"{table}.{ID_COLUMNS[table]} doesn't number itself "
                            "(it needs GENERATED ... AS IDENTITY or SERIAL)")
        for name, col in columns.items():
            if (name not in needed and col.get("is_nullable") == "NO" and not col.get("column_default")
                    and col.get("is_identity") != "YES"):
                problems.append(f"{table}.{name} is required but the app never fills it in "
                                "(give it a default or allow NULL)")
    return problems
