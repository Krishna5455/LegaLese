export type Document = {
  id: string;
  user_id: string;
  filename: string;
  document_type?: string | null;
  mime_type?: string | null;
  size_bytes?: number | null;
  storage_path?: string | null;
  status?: string | null;
  created_at: string;
  updated_at?: string | null;
};

export function getDocumentLabel(document: Document): string {
  return document.filename || `Document ${document.id.slice(0, 8)}`;
}

