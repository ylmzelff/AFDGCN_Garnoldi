import { useState, type FormEvent } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { TrafficCone, Eye, EyeOff, Loader2 } from 'lucide-react';
import { login } from '@/api/auth';
import { useAuthStore } from '@/store/useAuthStore';

export default function LoginPage() {
  const navigate = useNavigate();
  const setAuth = useAuthStore((s) => s.setAuth);

  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPwd, setShowPwd] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const data = await login({ username, password });
      setAuth(data.access_token, username);
      navigate('/dashboard', { replace: true });
    } catch {
      setError('Kullanıcı adı veya parola hatalı.');
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
          <h2 className="text-lg font-semibold text-gray-800">Giriş Yap</h2>

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
              placeholder="demo"
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
                autoComplete="current-password"
                placeholder="••••••••"
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

          {/* Hata */}
          {error && (
            <p className="rounded-lg bg-red-50 px-3 py-2 text-sm text-red-600 ring-1 ring-red-200">
              {error}
            </p>
          )}

          {/* Giriş Butonu */}
          <button
            type="submit"
            disabled={loading}
            className="flex w-full items-center justify-center gap-2 rounded-lg bg-brand py-2.5 text-sm font-semibold text-white shadow hover:bg-brand-light disabled:opacity-60 transition"
          >
            {loading && <Loader2 className="h-4 w-4 animate-spin" />}
            {loading ? 'Giriş yapılıyor…' : 'Giriş Yap'}
          </button>

          <p className="text-center text-xs text-gray-400">
            Demo: <span className="font-semibold text-gray-600">demo / demo123</span>
          </p>

          <p className="text-center text-sm text-gray-500">
            Hesabın yok mu?{' '}
            <Link to="/register" className="font-semibold text-brand hover:underline">
              Hesap Oluştur
            </Link>
          </p>
        </form>
      </div>
    </div>
  );
}
