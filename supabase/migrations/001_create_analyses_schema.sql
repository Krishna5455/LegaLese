-- ============================================================
-- LegaLese Phase 5 — AI Contract Analysis Schema
-- Apply this migration in your Supabase SQL editor or via CLI.
-- ============================================================

-- ────────────────────────────────────────────────────────────
-- TABLE: analyses
-- One row per AI analysis run on a document.
-- ────────────────────────────────────────────────────────────
CREATE TABLE analyses (
  id              uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id     uuid        NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  user_id         uuid        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,

  -- Analysis lifecycle
  status          text        NOT NULL DEFAULT 'pending'
                                CHECK (status IN ('pending', 'analyzing', 'complete', 'failed')),
  error_message   text,

  -- Top-level AI output
  summary         text,

  -- Overall risk is computed deterministically by application logic
  -- from the per-finding risk levels. The AI does NOT set this value.
  overall_risk    text        CHECK (overall_risk IN ('low', 'medium', 'high', 'critical') OR overall_risk IS NULL),

  -- Whether the document text was truncated before sending to the AI
  was_truncated   boolean     NOT NULL DEFAULT false,

  -- AI model metadata (for auditability and cost tracking)
  model_used      text,
  input_tokens    integer,
  output_tokens   integer,

  -- Timestamps
  analyzed_at     timestamptz,
  created_at      timestamptz NOT NULL DEFAULT now(),
  updated_at      timestamptz NOT NULL DEFAULT now()
);

-- ────────────────────────────────────────────────────────────
-- TABLE: findings
-- AI-identified risks/issues in the contract.
-- Every finding is traceable to source text in the document.
-- ────────────────────────────────────────────────────────────
CREATE TABLE findings (
  id              uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  analysis_id     uuid        NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
  document_id     uuid        NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  user_id         uuid        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,

  -- Finding content
  category        text        NOT NULL,
  risk_level      text        NOT NULL
                                CHECK (risk_level IN ('info', 'low', 'medium', 'high', 'critical')),
  explanation     text        NOT NULL,
  why_it_matters  text,

  -- Evidence / source reference
  evidence_text   text,
  source_section  text,
  page_number     integer,
  section_index   integer,

  -- Confidence expressed by the AI for this finding
  confidence      text        CHECK (confidence IN ('low', 'medium', 'high') OR confidence IS NULL),

  sort_order      integer     NOT NULL DEFAULT 0,
  created_at      timestamptz NOT NULL DEFAULT now()
);

-- ────────────────────────────────────────────────────────────
-- TABLE: key_terms
-- ────────────────────────────────────────────────────────────
CREATE TABLE key_terms (
  id              uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  analysis_id     uuid        NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
  document_id     uuid        NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  user_id         uuid        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,

  term            text        NOT NULL,
  definition      text        NOT NULL,
  source_section  text,
  page_number     integer,
  section_index   integer,

  sort_order      integer     NOT NULL DEFAULT 0,
  created_at      timestamptz NOT NULL DEFAULT now()
);

-- ────────────────────────────────────────────────────────────
-- TABLE: obligations
-- ────────────────────────────────────────────────────────────
CREATE TABLE obligations (
  id              uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  analysis_id     uuid        NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
  document_id     uuid        NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  user_id         uuid        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,

  party           text,
  description     text        NOT NULL,
  source_section  text,
  page_number     integer,
  section_index   integer,

  sort_order      integer     NOT NULL DEFAULT 0,
  created_at      timestamptz NOT NULL DEFAULT now()
);

-- ────────────────────────────────────────────────────────────
-- TABLE: questions
-- ────────────────────────────────────────────────────────────
CREATE TABLE questions (
  id              uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  analysis_id     uuid        NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
  document_id     uuid        NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  user_id         uuid        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,

  question_text   text        NOT NULL,
  context         text,

  sort_order      integer     NOT NULL DEFAULT 0,
  created_at      timestamptz NOT NULL DEFAULT now()
);

-- ────────────────────────────────────────────────────────────
-- INDEXES
-- ────────────────────────────────────────────────────────────
CREATE INDEX analyses_document_id_idx    ON analyses(document_id);
CREATE INDEX analyses_user_id_idx        ON analyses(user_id);
CREATE INDEX analyses_status_idx         ON analyses(status);
CREATE INDEX findings_analysis_id_idx    ON findings(analysis_id);
CREATE INDEX findings_document_id_idx    ON findings(document_id);
CREATE INDEX key_terms_analysis_id_idx   ON key_terms(analysis_id);
CREATE INDEX obligations_analysis_id_idx ON obligations(analysis_id);
CREATE INDEX questions_analysis_id_idx   ON questions(analysis_id);

-- ────────────────────────────────────────────────────────────
-- updated_at TRIGGER
-- ────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION handle_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER analyses_updated_at
  BEFORE UPDATE ON analyses
  FOR EACH ROW EXECUTE FUNCTION handle_updated_at();

-- ────────────────────────────────────────────────────────────
-- ROW LEVEL SECURITY
-- ────────────────────────────────────────────────────────────
ALTER TABLE analyses    ENABLE ROW LEVEL SECURITY;
ALTER TABLE findings    ENABLE ROW LEVEL SECURITY;
ALTER TABLE key_terms   ENABLE ROW LEVEL SECURITY;
ALTER TABLE obligations ENABLE ROW LEVEL SECURITY;
ALTER TABLE questions   ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can manage their own analyses"
  ON analyses FOR ALL
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can manage their own findings"
  ON findings FOR ALL
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can manage their own key terms"
  ON key_terms FOR ALL
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can manage their own obligations"
  ON obligations FOR ALL
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can manage their own questions"
  ON questions FOR ALL
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);
