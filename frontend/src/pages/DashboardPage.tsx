import { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { LogOut, RefreshCw, TrafficCone, Zap, ChevronRight, Activity, Wifi, WifiOff, Brain, BarChart2 } from 'lucide-react';
import { useAuthStore } from '@/store/useAuthStore';
import { usePhaseStore } from '@/store/usePhaseStore';
import { useWebSocket } from '@/hooks/useWebSocket';
import { apiClient } from '@/api/client';
import { junctionName, junctionArms } from '@/config/junctionMeta';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend,
} from 'recharts';

// ─── Bölge & Kavşak tanımları ────────────────────────────────────────────

interface JunctionDef {
  id: number;
  name: string;
  arms: readonly string[];
}

interface RegionDef {
  name: string;
  label: string;
  city: string;
  description: string;
  useModel: boolean;
  junctions: JunctionDef[];
}

type LoadLevel = 'low' | 'medium' | 'high' | null;

interface JunctionLive {
  totalVehicles: number;
  maxArmVehicles: number;
  load: LoadLevel;
}

interface ChartPoint { time: string; real: number | null; predicted: number | null }

interface PhaseArm {
  arm: string; name: string; vehicle_count: number;
  green: number; yellow: number; red: number;
  cycle_time: number; status: string;
}

function loadLevel(total: number): LoadLevel {
  if (total === 0) return null;
  if (total < 40)  return 'low';
  if (total < 100) return 'medium';
  return 'high';
}

// ─── Dashboard ─────────────────────────────────────────────────────────────
export default function DashboardPage() {
  const navigate = useNavigate();
  const { username, logout } = useAuthStore();
  const { wsStatus } = usePhaseStore();
  useWebSocket();

  // Bölge listesi — API'den dinamik yüklenir
  const [regions, setRegions] = useState<RegionDef[]>([]);
  const [selectedRegion, setSelectedRegion] = useState<RegionDef | null>(null);

  // Kavşak canlı verileri (kart badge'leri için)
  const [liveData, setLiveData] = useState<Record<number, JunctionLive>>({});
  const [liveLoading, setLiveLoading] = useState(true);

  const [selectedJunction, setSelectedJunction] = useState<JunctionDef | null>(null);
  const [selectedArm, setSelectedArm] = useState<string | null>(null);
  const [chartData, setChartData] = useState<ChartPoint[]>([]);
  const [source, setSource] = useState('');
  const [kayseriOk, setKayseriOk] = useState<boolean | null>(null);
  const [phaseData, setPhaseData] = useState<PhaseArm[] | null>(null);
  const [phaseCycle, setPhaseCycle] = useState(60);
  const [phaseLoading, setPhaseLoading] = useState(false);

  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Bölge listesini API'den yükle
  useEffect(() => {
    apiClient.get('/predict/regions').then((res) => {
      type ApiRegion = { name: string; city: string; description: string; junction_ids: number[]; use_model: boolean };
      const data: ApiRegion[] = res.data?.regions ?? [];
      const built = data.map((r): RegionDef => ({
        name: r.name,
        label: r.name.toUpperCase(),
        city: r.city,
        description: r.description ?? `${r.junction_ids.length} kavşak`,
        useModel: r.use_model,
        junctions: r.junction_ids.map((id) => ({
          id,
          name: junctionName(id),
          arms: junctionArms(id),
        })),
      }));
      setRegions(built);
      setSelectedRegion(built[0] ?? null);
    }).catch(() => { /* API hazır değilse sessizce geç */ });
  }, []);

  // Bölge verisi çekip kart badge'lerini güncelle
  const fetchLiveData = useCallback(async (regionName?: string) => {
    const name = regionName ?? selectedRegion?.name;
    if (!name) return;
    try {
      const res = await apiClient.post(`/predict/region/${name}`);
      const predictions: Record<string, Record<string, number>> = res.data.predictions ?? {};
      const next: Record<number, JunctionLive> = {};
      for (const [jidStr, arms] of Object.entries(predictions)) {
        const jid = Number(jidStr);
        const vals = Object.values(arms as Record<string, number>);
        const total = vals.reduce((a, b) => a + b, 0);
        const max   = vals.length ? Math.max(...vals) : 0;
        next[jid] = { totalVehicles: Math.round(total), maxArmVehicles: Math.round(max), load: loadLevel(total) };
      }
      setLiveData(next);
    } catch { /* sessizce geç */ }
    finally { setLiveLoading(false); }
  }, [selectedRegion?.name]);

  // Sayfa açılınca 1 kez + her 60s yenile
  useEffect(() => {
    if (!selectedRegion) return;
    setLiveLoading(true);
    fetchLiveData();
    const id = setInterval(fetchLiveData, 60_000);
    return () => clearInterval(id);
  }, [fetchLiveData, selectedRegion]);

  const stopPolling = useCallback(() => {
    if (pollingRef.current) { clearInterval(pollingRef.current); pollingRef.current = null; }
  }, []);

  const fetchPoint = useCallback(async (junctionId: number, arm: string, initial = false) => {
    try {
      const res = await apiClient.post(`/predict/region/${selectedRegion?.name}`);
      const data = res.data;
      const predicted = data.predictions?.[junctionId]?.[arm] ?? null;
      setSource(data.source ?? '');
      setKayseriOk(data.kayseri_ok ?? null);

      if (initial) {
        // Zaman serisi ile tüm geçmişi doldurup son noktaya tahmini ekle
        const series: number[] = data.time_series?.[junctionId]?.[arm] ?? [];
        const n = series.length;
        const now = Date.now();
        const SLOT_MS = 10 * 60 * 1000; // 10 dakika
        const points: ChartPoint[] = series.map((val, i) => {
          const offsetMs = (i - (n - 1)) * SLOT_MS;
          const t = new Date(now + offsetMs);
          return {
            time: t.toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' }),
            real: Math.round(val),
            predicted: i === n - 1 && predicted !== null ? Math.round(predicted) : null,
          };
        });
        setChartData(points);
      } else {
        // 10 dakikada bir yeni nokta ekle
        const real = data.raw_data?.[junctionId]?.[arm] ?? null;
        setChartData(prev => {
          const point: ChartPoint = {
            time: new Date().toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit' }),
            real: real !== null ? Math.round(real) : null,
            predicted: predicted !== null ? Math.round(predicted) : null,
          };
          return [...prev, point].slice(-40);
        });
      }
    } catch { /* sunucu hatası sessizce geç */ }
  }, [selectedRegion?.name]);

  // Junction veya kol değişince polling sıfırla
  useEffect(() => {
    stopPolling();
    if (!selectedJunction || !selectedArm) return;
    setChartData([]);
    fetchPoint(selectedJunction.id, selectedArm, true);  // ilk yükleme: tam zaman serisi
    pollingRef.current = setInterval(
      () => fetchPoint(selectedJunction.id, selectedArm, false),
      600_000  // 10 dakika
    );
    return stopPolling;
  }, [selectedJunction?.id, selectedArm, fetchPoint, stopPolling]);

  async function handleGetPhase() {
    if (!selectedJunction) return;
    setPhaseLoading(true);
    setPhaseData(null);
    try {
      const res = await apiClient.post(`/predict/junction/${selectedJunction.id}?region=${selectedRegion?.name}`);
      const detail = res.data;
      const cycleTime: number = detail.phase_recommendation?.cycle_time ?? 60;
      setPhaseCycle(cycleTime);
      const arms: PhaseArm[] = Object.entries(detail.arms ?? {}).map(
        ([arm, d]: [string, any]) => ({
          arm, name: d.name ?? `Kol ${arm}`,
          vehicle_count: d.vehicle_count ?? 0,
          green: d.green ?? 0, yellow: d.yellow ?? 3, red: d.red ?? 0,
          cycle_time: cycleTime, status: d.status ?? 'low',
        }),
      );
      setPhaseData(arms);
    } catch { /* hata sessizce geç */ }
    finally { setPhaseLoading(false); }
  }

  function handleRegionSelect(region: RegionDef) {
    if (region.name === selectedRegion?.name) return;
    stopPolling();
    setSelectedRegion(region);
    setSelectedJunction(null);
    setSelectedArm(null);
    setChartData([]);
    setPhaseData(null);
    setLiveData({});
    setLiveLoading(true);
    fetchLiveData(region.name);
  }

  function handleJunctionSelect(j: JunctionDef) {
    setSelectedJunction(j); setSelectedArm(null); setChartData([]); setPhaseData(null);
  }

  function handleLogout() { stopPolling(); logout(); navigate('/login', { replace: true }); }

  return (
    <div className="min-h-screen bg-gray-50">

      {/* ── Navbar ───────────────────────────────────────────────────── */}
      <header className="sticky top-0 z-20 bg-brand shadow-lg">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3">
          <div className="flex items-center gap-3">
            <TrafficCone className="h-6 w-6 text-white" />
            <div>
              <h1 className="text-base font-bold text-white leading-tight">Phase API</h1>
              <p className="text-xs text-white/50 leading-tight">{selectedRegion?.city?.toUpperCase() ?? '—'} · {selectedRegion?.label ?? ''} Bölgesi</p>
            </div>
          </div>
          <div className="hidden sm:flex items-center gap-3">
            {kayseriOk !== null && (
              kayseriOk
                ? <Wifi className="h-4 w-4 text-green-300" />
                : <WifiOff className="h-4 w-4 text-yellow-300" />
            )}
            <span className="text-xs text-white/50">
              {wsStatus === 'connected' ? '● Canlı' : wsStatus === 'connecting' ? '○ Bağlanıyor' : '● Çevrimdışı'}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className="hidden text-sm text-white/60 sm:block">{username}</span>
            <button
              onClick={handleLogout}
              className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm text-white/80 hover:bg-white/10 hover:text-white transition"
            >
              <LogOut className="h-4 w-4" />
              <span className="hidden sm:inline">Çıkış</span>
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-6 space-y-6">

        {/* ── Breadcrumb ───────────────────────────────────────────────── */}
        <nav className="flex items-center gap-2 text-sm text-gray-500">
          <span className="font-medium text-gray-700">{selectedRegion?.label ?? '—'}</span>
          {selectedJunction && (
            <><ChevronRight className="h-4 w-4" />
              <span className="font-semibold text-brand">{selectedJunction.name} #{selectedJunction.id}</span></>
          )}
          {selectedArm && (
            <><ChevronRight className="h-4 w-4" />
              <span className="font-semibold text-brand">Kol {selectedArm}</span></>
          )}
        </nav>

        {/* ── Bölge Seç ─────────────────────────────────────────────── */}
        <section>
          <h2 className="mb-3 text-xs font-bold uppercase tracking-widest text-gray-400">Bölge Seç</h2>
          <div className="flex gap-3 flex-wrap">
            {regions.map((r) => {
              const active = selectedRegion?.name === r.name;
              return (
                <button
                  key={r.name}
                  onClick={() => handleRegionSelect(r)}
                  className={`flex items-center gap-3 rounded-xl border-2 px-5 py-3 transition focus:outline-none ${
                    active
                      ? 'border-brand bg-brand text-white shadow-lg'
                      : 'border-gray-200 bg-white hover:border-brand/50 text-gray-700'
                  }`}
                >
                  <div className={`flex h-9 w-9 items-center justify-center rounded-lg ${
                    active ? 'bg-white/20' : 'bg-gray-100'
                  }`}>
                    {r.useModel
                      ? <Brain className={`h-5 w-5 ${active ? 'text-white' : 'text-brand'}`} />
                      : <BarChart2 className={`h-5 w-5 ${active ? 'text-white' : 'text-purple-500'}`} />
                    }
                  </div>
                  <div className="text-left">
                    <div className={`text-sm font-bold ${active ? 'text-white' : 'text-gray-800'}`}>{r.label}</div>
                    <div className={`text-[11px] ${active ? 'text-white/70' : 'text-gray-400'}`}>
                      {r.description} · {r.useModel ? 'AFDGCN' : 'Moving Avg'}
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        </section>

        {/* ── Adım 1: Kavşak Seç ──────────────────────────────────────── */}
        <section>
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-xs font-bold uppercase tracking-widest text-gray-400">Kavşak Seç — {selectedRegion?.label ?? ''}</h2>
            <div className="flex items-center gap-3 text-xs text-gray-400">
              <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-green-400 inline-block" />Düşük</span>
              <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-yellow-400 inline-block" />Orta</span>
              <span className="flex items-center gap-1"><span className="h-2 w-2 rounded-full bg-red-400 inline-block" />Yoğun</span>
              <button onClick={() => fetchLiveData()} className="ml-1 flex items-center gap-1 hover:text-gray-600 transition">
                <RefreshCw className="h-3.5 w-3.5" />Güncelle
              </button>
            </div>
          </div>
          <div className="grid grid-cols-3 gap-3 sm:grid-cols-5">
            {(selectedRegion?.junctions ?? []).map((j) => {
              const sel  = selectedJunction?.id === j.id;
              const live = liveData[j.id];
              const lvl  = live?.load ?? null;
              const dotColor =
                lvl === 'high'   ? 'bg-red-400' :
                lvl === 'medium' ? 'bg-yellow-400' :
                lvl === 'low'    ? 'bg-green-400' :
                                   'bg-gray-300';
              const barPct = live
                ? Math.min(100, Math.round((live.totalVehicles / 200) * 100))
                : 0;
              return (
                <button
                  key={j.id}
                  onClick={() => handleJunctionSelect(j)}
                  className={`group rounded-xl border-2 p-3 text-left transition focus:outline-none ${
                    sel
                      ? 'border-brand bg-brand text-white shadow-lg scale-[1.02]'
                      : 'border-gray-200 bg-white hover:border-brand/50 hover:shadow-md'
                  }`}
                >
                  {/* Üst satır: ID + canlı nokta */}
                  <div className="flex items-center justify-between mb-1.5">
                    <div className="flex items-center gap-1">
                      <TrafficCone className={`h-3.5 w-3.5 ${sel ? 'text-white/60' : 'text-brand'}`} />
                      <span className={`text-[11px] font-medium ${sel ? 'text-white/60' : 'text-gray-400'}`}>#{j.id}</span>
                    </div>
                    {liveLoading
                      ? <span className="h-2 w-2 rounded-full bg-gray-300 animate-pulse" />
                      : <span className={`h-2 w-2 rounded-full ${dotColor} ${!sel && lvl === 'high' ? 'animate-pulse' : ''}`} />
                    }
                  </div>
                  {/* İsim */}
                  <p className={`text-sm font-bold leading-tight ${sel ? 'text-white' : 'text-gray-800'}`}>{j.name}</p>
                  {/* Araç sayısı */}
                  {!liveLoading && live ? (
                    <p className={`text-[11px] mt-0.5 font-semibold ${sel ? 'text-white/70' : 'text-gray-500'}`}>
                      {live.totalVehicles} araç
                    </p>
                  ) : (
                    <p className={`text-[11px] mt-0.5 ${sel ? 'text-white/50' : 'text-gray-400'}`}>{j.arms.length} kol</p>
                  )}
                  {/* Yük bar */}
                  {!sel && (
                    <div className="mt-2 h-1 w-full rounded-full bg-gray-100 overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all ${
                          lvl === 'high' ? 'bg-red-400' : lvl === 'medium' ? 'bg-yellow-400' : 'bg-green-400'
                        }`}
                        style={{ width: `${barPct}%` }}
                      />
                    </div>
                  )}
                </button>
              );
            })}
          </div>
        </section>

        {/* ── Adım 2: Faz Önerisi ─────────────────────────────────────── */}
        {selectedJunction && (
          <section className="animate-fade-in">
            <div className="flex items-center gap-4 flex-wrap">
              <button
                onClick={handleGetPhase}
                disabled={phaseLoading}
                className="flex items-center gap-2 rounded-xl bg-brand px-6 py-3 text-white font-semibold shadow-md hover:bg-brand/90 disabled:opacity-60 transition"
              >
                <Zap className="h-5 w-5" />
                {phaseLoading ? 'Hesaplanıyor…' : `${selectedJunction.name} için Faz Önerisi Al`}
              </button>
              {phaseData && (
                <span className="text-xs text-gray-400 flex items-center gap-1">
                  Döngü süresi: <strong className="text-gray-700">{phaseCycle}s</strong>
                </span>
              )}
            </div>

            {phaseData && (
              <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                {phaseData.map((arm) => (
                  <div key={arm.arm} className="rounded-2xl border border-gray-100 bg-white p-4 shadow-sm">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xl font-bold text-gray-800">Kol {arm.arm}</span>
                      <span className={`text-xs rounded-full px-2 py-0.5 font-semibold ${
                        arm.status === 'low'    ? 'bg-green-100 text-green-700' :
                        arm.status === 'medium' ? 'bg-yellow-100 text-yellow-700' :
                                                  'bg-red-100 text-red-700'
                      }`}>
                        {arm.status === 'low' ? 'Düşük' : arm.status === 'medium' ? 'Orta' : 'Yoğun'}
                      </span>
                    </div>
                    <p className="text-[11px] text-gray-400 mb-1 truncate" title={arm.name}>{arm.name}</p>
                    <p className="text-sm text-gray-600 font-medium mb-3">{Math.round(arm.vehicle_count)} araç</p>
                    <div className="space-y-1.5">
                      <PhaseBar label="Yeşil"   seconds={arm.green}  total={phaseCycle} color="bg-green-500" />
                      <PhaseBar label="Sarı"    seconds={arm.yellow} total={phaseCycle} color="bg-yellow-400" />
                      <PhaseBar label="Kırmızı" seconds={arm.red}    total={phaseCycle} color="bg-red-500" />
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>
        )}

        {/* ── Adım 3: Kol Seç ─────────────────────────────────────────── */}
        {selectedJunction && (
          <section className="animate-fade-in">
            <h2 className="mb-3 text-xs font-bold uppercase tracking-widest text-gray-400">
              {selectedJunction.name} — Kol Seç (Trafik Grafiği)
            </h2>
            <div className="flex gap-3 flex-wrap">
              {selectedJunction.arms.map((arm) => {
                const live = liveData[selectedJunction.id];
                const approxCount = live ? Math.round(live.totalVehicles / selectedJunction.arms.length) : null;
                const isSelected = selectedArm === arm;
                return (
                  <button
                    key={arm}
                    onClick={() => setSelectedArm(arm)}
                    className={`min-w-[96px] rounded-xl border-2 px-5 py-3 text-center transition focus:outline-none ${
                      isSelected
                        ? 'border-brand bg-brand text-white shadow-lg'
                        : 'border-gray-200 bg-white hover:border-brand/40 text-gray-700'
                    }`}
                  >
                    <div className="text-base font-bold">Kol {arm}</div>
                    {approxCount !== null && (
                      <div className={`text-[11px] mt-0.5 ${isSelected ? 'text-white/70' : 'text-gray-400'}`}>
                        ~{approxCount} araç
                      </div>
                    )}
                  </button>
                );
              })}
            </div>
          </section>
        )}

        {/* ── Adım 4: Grafik ──────────────────────────────────────────── */}
        {selectedJunction && selectedArm && (
          <section className="animate-fade-in space-y-4">
            <div className="flex items-center justify-between flex-wrap gap-2">
              <div>
                <h2 className="text-base font-bold text-gray-800">
                  {selectedJunction.name} — Kol {selectedArm} — Araç Akışı
                </h2>
                <p className="text-xs text-gray-400 mt-0.5 flex items-center gap-1">
                  {source && <><Activity className="h-3 w-3" />{source} · </>}
                  10dk yenileme
                </p>
              </div>
              <button
                onClick={() => fetchPoint(selectedJunction.id, selectedArm, true)}
                className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm text-gray-600 border border-gray-200 bg-white hover:bg-gray-50 transition"
              >
                <RefreshCw className="h-4 w-4" />Yenile
              </button>
            </div>

            <div className="rounded-2xl bg-white border border-gray-100 shadow-sm p-5">
              {chartData.length === 0 ? (
                <div className="flex h-52 items-center justify-center text-gray-400 text-sm gap-2">
                  <RefreshCw className="h-4 w-4 animate-spin" /> Veri yükleniyor…
                </div>
              ) : (
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis dataKey="time" tick={{ fontSize: 10 }} interval="preserveStartEnd" />
                    <YAxis tick={{ fontSize: 11 }} unit=" araç" width={60} />
                    <Tooltip
                      contentStyle={{ borderRadius: 8, border: '1px solid #e5e7eb', fontSize: 12 }}
                      formatter={(v: number) => [`${v} araç`]}
                    />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <Line
                      type="monotone" dataKey="real" name="Gerçek"
                      stroke="#3b82f6" strokeWidth={2.5} dot={false} activeDot={{ r: 4 }} connectNulls
                    />
                    <Line
                      type="monotone" dataKey="predicted" name="Tahmin (AFDGCN)"
                      stroke="#f97316" strokeWidth={2.5} strokeDasharray="7 3"
                      dot={false} activeDot={{ r: 4 }} connectNulls
                    />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          </section>
        )}


      </main>
    </div>
  );
}

// ─── PhaseBar yardımcı bileşeni ────────────────────────────────────────────
function PhaseBar({ label, seconds, total, color }: {
  label: string; seconds: number; total: number; color: string;
}) {
  const pct = total > 0 ? Math.min(100, Math.round((seconds / total) * 100)) : 0;
  return (
    <div className="flex items-center gap-2">
      <div className={`h-2.5 w-2.5 flex-shrink-0 rounded-full ${color}`} />
      <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
        <div className={`h-full ${color} rounded-full transition-all`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-gray-500 w-8 text-right font-mono">{seconds}s</span>
    </div>
  );
}
