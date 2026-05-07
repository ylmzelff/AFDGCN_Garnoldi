import { useState, type FormEvent } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { TrafficCone, Eye, EyeOff, Loader2 } from 'lucide-react';
import { register } from '@/api/auth';
import { useAuthStore } from '@/store/useAuthStore';

export default function RegisterPage() {
  const navigate = useNavigate();
  const setAuth = useAuthStore((s) => s.setAuth);

  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm]   = useState('');
  const [showPwd, setShowPwd]   = useState(false);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState<string | null>(null);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);

    if (username.trim().length < 3) {
      setError('Kullanıcı adı en az 3 karakter olmalıdır.');
      return;
    }
    if (password.length < 6) {
      setError('Parola en az 6 karakter olmalıdır.');
      return;
    }
    if (password !== confirm) {
      setError('Parola ve onay eşleşmiyor.');
      return;
    }

    setLoading(true);
    try {
      const data = await register({ username: username.trim(), password });
      setAuth(data.access_token, username.trim());
      navigate('/dashboard', { replace: true });
    } catch (err: any) {
      const detail = err?.response?.data?.detail;
      setError(
        typeof detail === 'string'
          ? detail
          : 'Hesap oluşturulamadı. Lütfen tekrar deneyin.'
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-brand to-slate-800 px-4">
      <div className="w-full max-w-sm">
        {/* Logo */}
        <div className="mb-8 flex flex-col items-center gap-3 text-white">
          <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-white/10 backdrop-blur ring-1 ring-white/20 shadow-xl">
            <TrafficCone className="h-8 w-8 text-white" />
          </div>
          <div className="text-center">
            <h1 className="text-2xl font-bold tracking-tight">Phase API</h1>
            <p className="text-sm text-white/60 mt-0.5">
              Kayseri Trafik Faz Tahmin Sistemi
            </p>
          </div>
        </div>

        {/* Kart */}
        <form
          onSubmit={handleSubmit}
          className="rounded-2xl bg-white p-8 shadow-2xl space-y-5"
        >
          <h2 className="text-lg font-semibold text-gray-800">Hesap Oluştur</h2>

          {/* Kullanıcı Adı */}
          <div>
            <label className="mb-1.5 block text-sm font-medium text-gray-600">
              Kullanıcı Adı
            </label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
              autoComplete="username"
              placeholder="en az 3 karakter"
              className="w-full rounded-lg border border-gray-300 px-3.5 py-2.5 text-sm outline-none transition focus:border-brand-light focus:ring-2 focus:ring-brand-light/20"
            />
          </div>

          {/* Parola */}
          <div>
            <label className="mb-1.5 block text-sm font-medium text-gray-600">
              Parola
            </label>
            <div className="relative">
              <input
                type={showPwd ? 'text' : 'password'}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                autoComplete="new-password"
                placeholder="en az 6 karakter"
                className="w-full rounded-lg border border-gray-300 px-3.5 py-2.5 pr-10 text-sm outline-none transition focus:border-brand-light focus:ring-2 focus:ring-brand-light/20"
              />
              <button
                type="button"
                onClick={() => setShowPwd((v) => !v)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                tabIndex={-1}
              >
                {showPwd ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
            </div>
          </div>

          {/* Parola Onay */}
          <div>
            <label className="mb-1.5 block text-sm font-medium text-gray-600">
              Parola Onay
            </label>
            <input
              type={showPwd ? 'text' : 'password'}
              value={confirm}
              onChange={(e) => setConfirm(e.target.value)}
              required
              autoComplete="new-password"
              placeholder="••••••••"
              className="w-full rounded-lg border border-gray-300 px-3.5 py-2.5 text-sm outline-none transition focus:border-brand-light focus:ring-2 focus:ring-brand-light/20"
            />
          </div>

          {/* Hata */}
          {error && (
            <p className="rounded-lg bg-red-50 px-3 py-2 text-sm text-red-600 ring-1 ring-red-200">
              {error}
            </p>
          )}

          {/* Gönder */}
          <button
            type="submit"
            disabled={loading}
            className="flex w-full items-center justify-center gap-2 rounded-lg bg-brand py-2.5 text-sm font-semibold text-white transition hover:bg-brand/90 disabled:opacity-60"
          >
            {loading && <Loader2 className="h-4 w-4 animate-spin" />}
            {loading ? 'Oluşturuluyor…' : 'Hesap Oluştur'}
          </button>

          {/* Giriş linki */}
          <p className="text-center text-sm text-gray-500">
            Zaten hesabın var mı?{' '}
            <Link
              to="/login"
              className="font-semibold text-brand hover:underline"
            >
              Giriş Yap
            </Link>
          </p>
        </form>
      </div>
    </div>
  );
}
