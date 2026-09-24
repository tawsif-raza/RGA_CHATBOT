IF DB_ID(N'enterprise_rag') IS NULL
BEGIN
    CREATE DATABASE enterprise_rag;
END;
GO

USE enterprise_rag;
GO

IF OBJECT_ID(N'dbo.documents', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.documents (
        document_id NVARCHAR(128) NOT NULL PRIMARY KEY,
        source_name NVARCHAR(512) NOT NULL,
        department NVARCHAR(128) NULL,
        role_name NVARCHAR(128) NULL,
        content NVARCHAR(MAX) NOT NULL,
        is_enabled BIT NOT NULL CONSTRAINT DF_documents_is_enabled DEFAULT 1,
        updated_at DATETIME2 NOT NULL CONSTRAINT DF_documents_updated_at DEFAULT SYSUTCDATETIME()
    );
END;
GO

IF OBJECT_ID(N'dbo.ingestion_state', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.ingestion_state (
        document_id NVARCHAR(128) NOT NULL PRIMARY KEY,
        content_hash CHAR(64) NOT NULL,
        ingested_at DATETIME2 NOT NULL CONSTRAINT DF_ingestion_state_ingested_at DEFAULT SYSUTCDATETIME(),
        CONSTRAINT FK_ingestion_state_documents FOREIGN KEY (document_id)
            REFERENCES dbo.documents(document_id)
    );
END;
GO
