export const ERRORS = {
  UNAUTHENTICATED: {
    status: 401,
    code: "UNAUTHENTICATED",
    message: "Kimlik doğrulaması gerekli.",
  },
  INVALID_TOKEN: {
    status: 401,
    code: "INVALID_TOKEN",
    message: "Geçersiz veya süresi dolmuş token.",
  },
  UNAUTHORIZED: {
    status: 403,
    code: "UNAUTHORIZED",
    message: "Bu işlem için yetkiniz yok.",
  },
  ACCOUNT_DISABLED: {
    status: 403,
    code: "ACCOUNT_DISABLED",
    message: "Hesap devre dışı.",
  },
  USER_NOT_FOUND: {
    status: 404,
    code: "USER_NOT_FOUND",
    message: "Kullanıcı bulunamadı.",
  },
  DUPLICATE: {
    status: 409,
    code: "DUPLICATE",
    message: "Bu kullanıcı adı zaten kullanılıyor.",
  },
  VALIDATION: {
    status: 422,
    code: "VALIDATION",
    message: "İstek doğrulama hatası.",
  },
  REGION_NOT_FOUND: {
    status: 404,
    code: "REGION_NOT_FOUND",
    message: "Bölge bulunamadı.",
  },
  SERVICE_UNAVAILABLE: {
    status: 503,
    code: "SERVICE_UNAVAILABLE",
    message: "Servis şu an kullanılamıyor.",
  },
  UNEXPECTED: {
    status: 500,
    code: "UNEXPECTED",
    message: "Beklenmedik bir hata oluştu.",
  },
} as const;

export type ErrorKey = keyof typeof ERRORS;
