-- ============================================================
-- DO NOT EXECUTE THIS MIGRATION
-- Historical / Reference Document Only
-- ============================================================
-- The tables below (analyses, clauses, findings, key_terms, obligations, reports)
-- already exist in the production Supabase database.
--
-- This file is kept in the repository purely for schema reference and documentation.
-- DO NOT run this against the remote Supabase project.
-- ============================================================

/*
-- Production Schema Overview:

CREATE TABLE analyses (
  id          uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id uuid        NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  user_id     uuid        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  risk_score  integer     NULL,
  summary     text        NULL,
  result      jsonb       NOT NULL,
  model       text        NOT NULL,
  created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE clauses (
  id            uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id   uuid        NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  section       text        NOT NULL,
  clause_number text        NULL,
  text          text        NOT NULL,
  page_number   integer     NULL,
  created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE findings (
  id             uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id    uuid        NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  clause_id      uuid        NULL REFERENCES clauses(id) ON DELETE SET NULL,
  risk_level     text        NOT NULL,
  category       text        NOT NULL,
  explanation    text        NOT NULL,
  why_it_matters text        NULL,
  questions      jsonb       NOT NULL DEFAULT '[]'::jsonb,
  confidence     numeric     NULL,
  created_at     timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE key_terms (
  id               uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id      uuid        NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  term             text        NOT NULL,
  value            text        NOT NULL,
  source_clause_id uuid        NULL REFERENCES clauses(id) ON DELETE SET NULL,
  created_at       timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE obligations (
  id                uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id       uuid        NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  description       text        NOT NULL,
  responsible_party text        NULL,
  deadline          text        NULL,
  source_clause_id  uuid        NULL REFERENCES clauses(id) ON DELETE SET NULL,
  created_at        timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE reports (
  id          uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id uuid        NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  user_id     uuid        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  file_path   text        NULL,
  created_at  timestamptz NOT NULL DEFAULT now()
);
*/
