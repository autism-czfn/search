-- Migration: Create evidence_cache table (P2) and update insights_cache (P8)
-- Run against the user database (mzhu_test_* prefix)

-- P2: Evidence cache table
CREATE TABLE IF NOT EXISTS evidence_cache (
    query_hash      TEXT NOT NULL,
    trigger_key     TEXT,
    language        TEXT NOT NULL DEFAULT 'en',
    time_bucket     DATE NOT NULL,
    query_text      TEXT,
    query_templates JSONB,
    retrieval_results JSONB,
    extracted_evidence JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at      TIMESTAMPTZ NOT NULL,
    UNIQUE (query_hash, language, time_bucket)
);

CREATE INDEX IF NOT EXISTS idx_evidence_cache_trigger
    ON evidence_cache (trigger_key)
    WHERE trigger_key IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_evidence_cache_expires
    ON evidence_cache (expires_at);

-- P8: Add cache_type column to insights cache for full vs base distinction
ALTER TABLE mzhu_test_insights_cache
    ADD COLUMN IF NOT EXISTS cache_type TEXT NOT NULL DEFAULT 'base';

-- Update unique constraint (if one exists on days alone, drop and recreate)
-- Note: run this only if needed — check existing constraints first
-- ALTER TABLE mzhu_test_insights_cache DROP CONSTRAINT IF EXISTS mzhu_test_insights_cache_days_key;
-- ALTER TABLE mzhu_test_insights_cache ADD CONSTRAINT mzhu_test_insights_cache_days_type_key UNIQUE (days, cache_type);
