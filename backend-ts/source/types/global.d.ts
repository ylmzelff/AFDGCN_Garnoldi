export interface CustomError extends Error {
  status?: number;
  code?: string;
  payload?: Record<string, unknown>;
}

export interface ApiErrorBody {
  error: true;
  code: string;
  message: string;
  details?: unknown;
}
