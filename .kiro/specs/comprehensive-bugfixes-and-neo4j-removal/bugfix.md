# Bugfix Requirements Document

## Introduction

This document specifies requirements for fixing multiple bugs in the enterprise RAG system and removing the Neo4j dependency. The bugs include deprecated datetime methods that will be removed in future Python versions, missing imports causing runtime errors, missing timestamps in error logs, and overly broad exception handling that masks specific errors. Additionally, the optional Neo4j integration adds unnecessary complexity even when not used, and should be completely removed to simplify deployment.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN datetime operations are performed in ingestion pipeline, connectors, and base classes THEN the system uses deprecated `datetime.utcnow()` method in 8+ locations

1.2 WHEN datetime operations are performed in DocsConnector THEN the system uses deprecated `datetime.utcfromtimestamp()` method in 2 locations

1.3 WHEN DOCS_PATH environment variable is configured and api.py line 219 attempts to instantiate DocsConnector THEN the system crashes with NameError because DocsConnector is not imported

1.4 WHEN log_error() method is called in logging.py THEN the system does not include a timestamp field in the log record, unlike log_query() and log_access_control()

1.5 WHEN Neo4j is not configured or unavailable THEN the system still includes Neo4j-related code, health checks, and complexity throughout the codebase

1.6 WHEN exceptions occur in multiple components THEN the system uses broad "except Exception" handlers in 20+ locations that can hide specific errors and make debugging difficult

### Expected Behavior (Correct)

2.1 WHEN datetime operations are performed anywhere in the codebase THEN the system SHALL use timezone-aware `datetime.now(timezone.utc)` instead of deprecated `datetime.utcnow()`

2.2 WHEN datetime operations convert timestamps in DocsConnector THEN the system SHALL use `datetime.fromtimestamp(ts, tz=timezone.utc)` instead of deprecated `datetime.utcfromtimestamp()`

2.3 WHEN DOCS_PATH environment variable is configured THEN the system SHALL successfully import and instantiate DocsConnector without runtime errors

2.4 WHEN log_error() method is called THEN the system SHALL include a timestamp field in the log record consistent with log_query() and log_access_control()

2.5 WHEN the system operates THEN the system SHALL not include any Neo4j-related code, imports, models, or health checks

2.6 WHEN exceptions occur in components THEN the system SHALL use specific exception types where appropriate to improve error visibility and debugging

### Unchanged Behavior (Regression Prevention)

3.1 WHEN datetime operations produce timezone-aware datetime objects THEN the system SHALL CONTINUE TO serialize them correctly to ISO format strings

3.2 WHEN DocsConnector processes files THEN the system SHALL CONTINUE TO extract correct modification times and create valid Document objects

3.3 WHEN ingestion pipeline runs THEN the system SHALL CONTINUE TO index documents into ChromaDB vector store successfully

3.4 WHEN structured logger emits log records THEN the system SHALL CONTINUE TO write valid JSON entries to the configured backend

3.5 WHEN retriever performs vector search THEN the system SHALL CONTINUE TO return relevant chunks based on semantic similarity

3.6 WHEN health endpoint is called THEN the system SHALL CONTINUE TO report status for ChromaDB, Ollama, and session store components

3.7 WHEN query endpoint processes requests THEN the system SHALL CONTINUE TO generate answers with citations and grounding scores

3.8 WHEN access controller filters content THEN the system SHALL CONTINUE TO enforce permission-based access control

3.9 WHEN conversation manager stores turns THEN the system SHALL CONTINUE TO maintain session history correctly

3.10 WHEN citation engine processes answers THEN the system SHALL CONTINUE TO extract and number citations from retrieved chunks
