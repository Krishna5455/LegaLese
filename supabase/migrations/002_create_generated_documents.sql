-- ============================================================
-- Migration: generated_documents table for CREATE workflow
-- ============================================================
-- Run this migration against your Supabase project if the
-- generated_documents table does not yet exist.
--
-- This table stores AI-generated legal document drafts with
-- structured sections for future explanation, review, and
-- customization features.
-- ============================================================

CREATE TABLE IF NOT EXISTS generated_documents (
  id                uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id           uuid        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  document_type     text        NOT NULL,
  title             text        NOT NULL,
  input_data        jsonb       NOT NULL,
  generated_content jsonb       NOT NULL,
  model             text        NULL,
  status            text        NOT NULL DEFAULT 'draft',
  created_at        timestamptz NOT NULL DEFAULT now(),
  updated_at        timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_generated_documents_user_id
  ON generated_documents (user_id);

CREATE INDEX IF NOT EXISTS idx_generated_documents_document_type
  ON generated_documents (document_type);

-- Row Level Security
ALTER TABLE generated_documents ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can select own generated documents"
  ON generated_documents
  FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own generated documents"
  ON generated_documents
  FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own generated documents"
  ON generated_documents
  FOR UPDATE
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own generated documents"
  ON generated_documents
  FOR DELETE
  USING (auth.uid() = user_id);

-- Optional: keep updated_at current on row updates
CREATE OR REPLACE FUNCTION update_generated_documents_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_generated_documents_updated_at ON generated_documents;

CREATE TRIGGER trg_generated_documents_updated_at
  BEFORE UPDATE ON generated_documents
  FOR EACH ROW
  EXECUTE FUNCTION update_generated_documents_updated_at();
