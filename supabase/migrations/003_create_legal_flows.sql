-- ============================================================
-- Migration: legal_flows table for AI-Powered LegalFlow Platform
-- ============================================================
-- Stores user legal objectives, dynamic workflow steps,
-- trust check statuses, and links to generated or analyzed documents.
-- ============================================================

CREATE TABLE IF NOT EXISTS legal_flows (
  id               uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id          uuid        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  title            text        NOT NULL,
  intent_query     text        NOT NULL,
  scenario_key     text        NOT NULL,
  current_stage    text        NOT NULL DEFAULT 'intent',
  status           text        NOT NULL DEFAULT 'active',
  document_id      uuid        NULL REFERENCES documents(id) ON DELETE SET NULL,
  generated_doc_id uuid        NULL REFERENCES generated_documents(id) ON DELETE SET NULL,
  steps            jsonb       NOT NULL DEFAULT '[]'::jsonb,
  trust_summary    jsonb       NOT NULL DEFAULT '{}'::jsonb,
  created_at       timestamptz NOT NULL DEFAULT now(),
  updated_at       timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_legal_flows_user_id
  ON legal_flows (user_id);

CREATE INDEX IF NOT EXISTS idx_legal_flows_status
  ON legal_flows (status);

CREATE INDEX IF NOT EXISTS idx_legal_flows_created_at
  ON legal_flows (created_at DESC);

-- Row Level Security
ALTER TABLE legal_flows ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can select own legal flows"
  ON legal_flows
  FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own legal flows"
  ON legal_flows
  FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own legal flows"
  ON legal_flows
  FOR UPDATE
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete own legal flows"
  ON legal_flows
  FOR DELETE
  USING (auth.uid() = user_id);

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_legal_flows_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_legal_flows_updated_at ON legal_flows;

CREATE TRIGGER trg_legal_flows_updated_at
  BEFORE UPDATE ON legal_flows
  FOR EACH ROW
  EXECUTE FUNCTION update_legal_flows_updated_at();
