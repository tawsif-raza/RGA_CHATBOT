"""Apply the MSSQL schema after the Docker service becomes healthy."""
from __future__ import annotations

from pathlib import Path

import pyodbc

from backend.config import Settings


def apply_schema(settings: Settings, script_path: Path) -> None:
    """Execute `GO`-separated T-SQL batches with autocommit enabled."""
    connection_string = f"DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={settings.mssql_host},{settings.mssql_port};UID={settings.mssql_user};PWD={settings.mssql_sa_password};TrustServerCertificate=yes;"
    with pyodbc.connect(connection_string, autocommit=True) as connection:
        cursor = connection.cursor()
        for batch in script_path.read_text(encoding="utf-8").split("\nGO\n"):
            if batch.strip():
                cursor.execute(batch)


if __name__ == "__main__":
    apply_schema(Settings(), Path(__file__).with_name("init.sql"))
