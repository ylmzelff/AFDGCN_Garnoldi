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
  Tooltip, Legend, ResponsiveContainer,
  PieChart, Pie, Cell,
} from 'recharts';
import { Fragment } from 'react';

const ARM_PHASE_COLORS = ['#3b82f6', '#22c55e', '#f97316', '#8b5cf6', '#06b6d4', '#ec4899'];

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
  /** Bu bölgenin veri kaynağının doğal zaman dilimi (dakika) — Kayseri: 10, Sivas: 60 */
  slotMinutes: number;
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
  if (total < 40) return 'low';
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
  const [phaseView, setPhaseView] = useState<'duration' | 'vehicles'>('duration');
  const [regionOpen, setRegionOpen] = useState(true);
  const [junctionOpen, setJunctionOpen] = useState(true);

  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Bölge listesini API'den yükle
  useEffect(() => {
    apiClient.get('/predict/regions').then((res) => {
      type ApiRegion = { name: string; city: string; description: string; junction_ids: number[]; use_model: boolean; slot_minutes?: number };
      const data: ApiRegion[] = res.data?.regions ?? [];
      const built = data.map((r): RegionDef => ({
        name: r.name,
        label: r.name.toUpperCase(),
        city: r.city,
        description: r.description ?? `${r.junction_ids.length} kavşak`,
        useModel: r.use_model,
        slotMinutes: r.slot_minutes ?? 10,
        junctions: r.junction_ids.map((id) => ({
          id,
          name: junctionName(r.city, id),
          arms: junctionArms(r.city, id),
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
      // time_series'den her kol için son bilinen (sıfır olmayan) gerçek değeri kullan
      const timeSeries: Record<string, Record<string, number[]>> = res.data.time_series ?? {};
      const next: Record<number, JunctionLive> = {};
      for (const [jidStr, arms] of Object.entries(timeSeries)) {
        const jid = Number(jidStr);
        let total = 0;
        let max = 0;
        for (const series of Object.values(arms as Record<string, number[]>)) {
          // Serinin sonundan ilk sıfır olmayan değeri bul
          const lastVal = [...series].reverse().find((v) => v > 0) ?? 0;
          total += lastVal;
          max = Math.max(max, lastVal);
        }
        next[jid] = { totalVehicles: Math.round(total), maxArmVehicles: Math.round(max), load: loadLevel(total) };
      }
      setLiveData(next);
    } catch { /* sessizce geç */ }
    finally { setLiveLoading(false); }
  }, [selectedRegion?.name]);

  // Sayfa açılınca 1 kez + bölgenin doğal slot sınırında yenile (Kayseri: 10dk, Sivas: 1sa)
  useEffect(() => {
    if (!selectedRegion) return;
    const slotMinutes = selectedRegion.slotMinutes;
    const slotMs = slotMinutes * 60_000;
    setLiveLoading(true);
    fetchLiveData();

    // Bir sonraki slot sınırına kaç ms kaldığını hesapla
    const msUntilNextSlot = () => {
      const now = new Date();
      const msLeft = ((slotMinutes - (now.getMinutes() % slotMinutes)) % slotMinutes) * 60_000
        - now.getSeconds() * 1000
        - now.getMilliseconds();
      return msLeft <= 0 ? slotMs : msLeft;
    };

    let intervalId: ReturnType<typeof setInterval> | null = null;
    const timeoutId = setTimeout(() => {
      fetchLiveData();                              // slot sınırında hemen güncelle
      intervalId = setInterval(fetchLiveData, slotMs); // sonra her slot'ta bir
    }, msUntilNextSlot());

    return () => {
      clearTimeout(timeoutId);
      if (intervalId) clearInterval(intervalId);
    };
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
      const slotMinutes: number = data.slot_minutes ?? selectedRegion?.slotMinutes ?? 10;

      // slot index → "HH:MM" (bölgenin doğal dilim uzunluğuna göre)
      const slotToTime = (slotIdx: number) => {
        const totalMinutes = slotIdx * slotMinutes;
        const hh = Math.floor(totalMinutes / 60);
        const mm = totalMinutes % 60;
        return `${String(hh).padStart(2, '0')}:${String(mm).padStart(2, '0')}`;
      };

      if (initial) {
        const series: number[] = data.time_series?.[junctionId]?.[arm] ?? [];
        const predSeries: number[] = data.prediction_series?.[junctionId]?.[arm] ?? [];
        const startSlot: number = data.time_series_start_slot ?? 0;

        // Baştaki sıfır (trafik yok) noktaları kırp
        let firstNonZero = 0;
        for (let i = 0; i < series.length; i++) {
          if (series[i] > 0) { firstNonZero = Math.max(0, i - 1); break; }
        }
        const trimmed = series.slice(firstNonZero);
        const trimmedPred = predSeries.slice(firstNonZero);
        const trimStart = startSlot + firstNonZero;

        const points: ChartPoint[] = trimmed.map((val, i) => ({
          time: slotToTime(trimStart + i),
          real: Math.round(val),
          predicted: trimmedPred[i] !== undefined ? Math.round(trimmedPred[i]) : null,
        }));
        setChartData(points);
      } else {
        // Bölgenin slot'unda bir yeni nokta ekle — tamamlanan (bir önceki) slotu göster
        const real = data.raw_data?.[junctionId]?.[arm] ?? null;
        const now = new Date();
        const slotsPerHour = 60 / slotMinutes;
        const curSlot = now.getHours() * slotsPerHour + Math.floor(now.getMinutes() / slotMinutes);
        const completedSlot = Math.max(0, curSlot - 1); // backend ile aynı mantık
        setChartData(prev => {
          const point: ChartPoint = {
            time: slotToTime(completedSlot),
            real: real !== null ? Math.round(real) : null,
            predicted: predicted !== null ? Math.round(predicted) : null,
          };
          return [...prev, point].slice(-(1440 / slotMinutes)); // max 1 gün
        });
      }
    } catch { /* sunucu hatası sessizce geç */ }
  }, [selectedRegion?.name, selectedRegion?.slotMinutes]);

  // Junction veya kol değişince polling sıfırla
  useEffect(() => {
    stopPolling();
    if (!selectedJunction || !selectedArm) return;
    setChartData([]);
    fetchPoint(selectedJunction.id, selectedArm, true);  // ilk yükleme: tam zaman serisi
    const pollMs = (selectedRegion?.slotMinutes ?? 10) * 60_000;
    pollingRef.current = setInterval(
      () => fetchPoint(selectedJunction.id, selectedArm, false),
      pollMs
    );
    return stopPolling;
  }, [selectedJunction?.id, selectedArm, selectedRegion?.slotMinutes, fetchPoint, stopPolling]);

  async function handleGetPhase() {
    if (!selectedJunction) return;
    setPhaseLoading(true);
    setPhaseData(null);
    try {
      const res = await apiClient.post(`/predict/junction/${selectedJunction.id}?region=${selectedRegion?.name}`);
      const detail = res.data;
      // Backend yanıtı: detail.phases = { _cycle_time, _total_vehicles, A: {...}, B: {...}, ... }
      const phases: Record<string, any> = detail.phases ?? {};
      const cycleTime: number = (phases['_cycle_time'] as number) ?? 60;
      setPhaseCycle(cycleTime);
      const arms: PhaseArm[] = Object.entries(phases)
        .filter(([k, v]) => !k.startsWith('_') && v !== null && typeof v === 'object')
        .map(([arm, d]) => ({
          arm,
          name: d.arm_name ?? `Kol ${arm}`,
          vehicle_count: d.vehicle_count ?? 0,
          green: d.green ?? 0,
          yellow: d.yellow ?? 3,
          red: d.red ?? 0,
          cycle_time: cycleTime,
          status: d.status ?? 'low',
        }));
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
    setRegionOpen(false);
    setJunctionOpen(true);
  }

  function handleJunctionSelect(j: JunctionDef) {
    setSelectedJunction(j); setSelectedArm(null); setChartData([]); setPhaseData(null);
    setJunctionOpen(false);
  }

  function handleLogout() { stopPolling(); logout(); navigate('/login', { replace: true }); }

  return (
    <div className="min-h-screen bg-gray-50">

      {/* ── Navbar ───────────────────────────────────────────────────── */}
      <header className="sticky top-0 z-20 bg-brand shadow-lg">
        <div className="flex w-full items-center justify-between px-6 py-3">
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

      <main className="w-full px-6 py-6 space-y-6">

        {/* ── Breadcrumb ───────────────────────────────────────────────── */}
        <nav className="flex items-center gap-2 text-sm text-gray-500">
          <span className="font-medium text-gray-700">{selectedRegion?.label ?? '—'}</span>
          {selectedJunction && (
            <><ChevronRight className="h-4 w-4" />
              <span className="font-semibold text-brand">{selectedJunction.name} #{selectedJunction.id}</span></>
          )}
          {selectedArm && (
            <><ChevronRight className="h-4 w-4" />
              <span className="font-semibold text-brand">Faz {selectedArm}</span></>
          )}
        </nav>

        {/* ── Bölge Seç ─────────────────────────────────────────────── */}
        <section>
          <button
            onClick={() => setRegionOpen((v) => !v)}
            className="flex w-full items-center justify-between rounded-xl border border-gray-200 bg-white px-4 py-2.5 shadow-sm hover:bg-gray-50 transition"
          >
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold uppercase tracking-widest text-gray-500">Bölge Seç</span>
              {selectedRegion && !regionOpen && (
                <span className="text-xs font-semibold text-brand bg-brand/10 px-2 py-0.5 rounded-full">{selectedRegion.label}</span>
              )}
            </div>
            <ChevronRight className={`h-4 w-4 text-gray-400 transition-transform duration-200 ${regionOpen ? 'rotate-90' : ''}`} />
          </button>
          {regionOpen && (
            <div className="mt-2 flex flex-wrap gap-2">
              {regions.map((r) => {
                const active = selectedRegion?.name === r.name;
                return (
                  <button
                    key={r.name}
                    onClick={() => handleRegionSelect(r)}
                    className={`flex items-center gap-2 rounded-full border px-3.5 py-1.5 text-sm font-medium transition focus:outline-none ${active
                        ? 'border-brand bg-brand text-white shadow-md'
                        : 'border-gray-200 bg-white text-gray-700 hover:border-brand/50 hover:shadow-sm'
                      }`}
                  >
                    {r.useModel
                      ? <Brain className={`h-3.5 w-3.5 flex-shrink-0 ${active ? 'text-white/70' : 'text-brand'}`} />
                      : <BarChart2 className={`h-3.5 w-3.5 flex-shrink-0 ${active ? 'text-white/70' : 'text-purple-500'}`} />
                    }
                    <span>{r.label}</span>
                    <span className={`text-[11px] font-normal ${active ? 'text-white/60' : 'text-gray-400'}`}>
                      {r.useModel ? 'Garnoldi' : 'Moving Avg'}
                    </span>
                  </button>
                );
              })}
            </div>
          )}
        </section>

        {/* ── Adım 1: Kavşak Seç ──────────────────────────────────────── */}
        <section>
          <button
            onClick={() => setJunctionOpen((v) => !v)}
            className="flex w-full items-center justify-between rounded-xl border border-gray-200 bg-white px-4 py-2.5 shadow-sm hover:bg-gray-50 transition"
          >
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold uppercase tracking-widest text-gray-500">Kavşak Seç — {selectedRegion?.label ?? ''}</span>
              {selectedJunction && !junctionOpen && (
                <span className="text-xs font-semibold text-brand bg-brand/10 px-2 py-0.5 rounded-full">{selectedJunction.name}</span>
              )}
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={(e) => { e.stopPropagation(); fetchLiveData(); }}
                className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-600 transition px-2 py-1 rounded-lg hover:bg-gray-100"
              >
                <RefreshCw className="h-3.5 w-3.5" />Güncelle
              </button>
              <ChevronRight className={`h-4 w-4 text-gray-400 transition-transform duration-200 ${junctionOpen ? 'rotate-90' : ''}`} />
            </div>
          </button>

          {junctionOpen && (
            <>
              {/* Pill listesi */}
              <div className="mt-2 flex flex-wrap gap-2">
                {(selectedRegion?.junctions ?? []).map((j) => {
                  const sel = selectedJunction?.id === j.id;
                  const live = liveData[j.id];
                  return (
                    <button
                      key={j.id}
                      onClick={() => handleJunctionSelect(j)}
                      className={`flex items-center gap-2 rounded-full border px-3.5 py-1.5 text-sm font-medium transition focus:outline-none ${sel
                          ? 'border-brand bg-brand text-white shadow-md'
                          : 'border-gray-200 bg-white text-gray-700 hover:border-brand/50 hover:shadow-sm'
                        }`}
                    >
                      <TrafficCone className={`h-3.5 w-3.5 flex-shrink-0 ${sel ? 'text-white/70' : 'text-brand'}`} />
                      <span>{j.name}</span>
                      {!liveLoading && live && (
                        <span className={`text-[11px] font-normal ${sel ? 'text-white/60' : 'text-gray-400'}`}>
                          {live.totalVehicles} araç
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>

              {/* Seçili kavşak detay kartı */}
              {selectedJunction && (() => {
                const live = liveData[selectedJunction.id];
                return (
                  <div className="mt-2 flex items-center gap-4 rounded-2xl border border-brand/20 bg-brand/5 px-4 py-3">
                    <TrafficCone className="h-5 w-5 text-brand flex-shrink-0" />
                    <div className="flex-1 min-w-0">
                      <p className="font-bold text-gray-800 text-sm">{selectedJunction.name}</p>
                      <p className="text-[11px] text-gray-400">#{selectedJunction.id} · {selectedJunction.arms.length} kol</p>
                    </div>
                    {live && (
                      <div className="text-right flex-shrink-0">
                        <p className="text-lg font-bold text-brand">{live.totalVehicles}</p>
                        <p className="text-[11px] text-gray-400">araç</p>
                      </div>
                    )}
                  </div>
                );
              })()}
            </>
          )}
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
            </div>

            {phaseData && (
              <div className="mt-4 rounded-2xl border border-gray-100 bg-white shadow-sm overflow-hidden">

                {/* ── Header ── */}
                <div className="flex flex-wrap items-center justify-between gap-3 px-5 py-3 bg-gray-50 border-b border-gray-100">
                  <div className="flex items-center gap-2.5 flex-wrap">
                    <span className="font-semibold text-gray-800">{selectedJunction.name}</span>
                    <span className="text-xs bg-brand/10 text-brand font-bold px-2.5 py-0.5 rounded-full">{phaseCycle}s döngü</span>
                    <span className="text-xs bg-gray-200 text-gray-600 font-medium px-2 py-0.5 rounded-full">Webster Oranınlı Mod</span>
                    {source && (
                      <span className={`text-xs px-2 py-0.5 rounded-full font-semibold ${source === 'AFDGCN' ? 'bg-indigo-100 text-indigo-700' : 'bg-purple-100 text-purple-700'
                        }`}>{source === 'AFDGCN' ? 'Garnoldi' : source}</span>
                    )}
                  </div>
                  <div className="flex rounded-lg overflow-hidden border border-gray-200 text-xs">
                    <button
                      onClick={() => setPhaseView('duration')}
                      className={`px-3 py-1.5 font-medium transition ${phaseView === 'duration' ? 'bg-brand text-white' : 'text-gray-500 hover:bg-gray-100'
                        }`}
                    >Faz Süre Dağılımı</button>
                    <button
                      onClick={() => setPhaseView('vehicles')}
                      className={`px-3 py-1.5 font-medium border-l border-gray-200 transition ${phaseView === 'vehicles' ? 'bg-brand text-white' : 'text-gray-500 hover:bg-gray-100'
                        }`}
                    >Araç Sayısı Dağılımı</button>
                  </div>
                </div>

                {/* ── Body ── */}
                <div className="flex flex-col lg:flex-row">

                  {/* Left — Cycle timeline + arm bars */}
                  <div className="flex-1 p-5">

                    {/* Cycle stacked bar */}
                    <p className="text-[10px] font-bold uppercase tracking-widest text-gray-400 mb-2">Döngü Zaman Çizelgesi ({phaseCycle}s)</p>
                    <div className="flex h-9 rounded-lg overflow-hidden mb-5 ring-1 ring-black/5">
                      {phaseData.map((arm, i) => (
                        <Fragment key={arm.arm}>
                          <div
                            style={{ flex: arm.green, background: ARM_PHASE_COLORS[i % ARM_PHASE_COLORS.length] }}
                            title={`Kol ${arm.arm} Yeşil: ${arm.green}s`}
                            className="relative flex items-center justify-center"
                          >
                            {arm.green >= 10 && (
                              <span className="text-[10px] font-bold text-white/90 select-none">{arm.arm}</span>
                            )}
                          </div>
                          <div
                            style={{ flex: arm.yellow, background: '#eab308' }}
                            title={`Kol ${arm.arm} Sarı: ${arm.yellow}s`}
                          />
                          <div
                            style={{ flex: 6, background: '#334155' }}
                            title="Koruma: 6s"
                          />
                        </Fragment>
                      ))}
                    </div>

                    {/* Arm rows — tıklayınca trafik grafiği açılır */}
                    <p className="text-[10px] text-gray-400 mt-1 mb-1">Trafik grafiği için bir faza tıklayın</p>
                    <div className="space-y-2">
                      {phaseData.map((arm, i) => {
                        const color = ARM_PHASE_COLORS[i % ARM_PHASE_COLORS.length];
                        const maxGreen = Math.max(...phaseData.map((a) => a.green), 1);
                        const maxVeh = Math.max(...phaseData.map((a) => a.vehicle_count), 1);
                        const barPct = phaseView === 'duration'
                          ? (arm.green / maxGreen) * 100
                          : (arm.vehicle_count / maxVeh) * 100;
                        const label = phaseView === 'duration'
                          ? `${arm.green}s`
                          : `${Math.round(arm.vehicle_count)} araç`;
                        const isActive = selectedArm === arm.arm;
                        return (
                          <button
                            key={arm.arm}
                            onClick={() => setSelectedArm(isActive ? null : arm.arm)}
                            className={`w-full flex items-center gap-3 rounded-xl px-3 py-2 transition text-left border-2 ${isActive
                                ? 'border-opacity-100 shadow-sm'
                                : 'border-transparent hover:bg-gray-50'
                              }`}
                            style={isActive ? { borderColor: color, background: color + '12' } : {}}
                          >
                            <div className="w-12 text-sm font-bold flex-shrink-0" style={{ color }}>Faz {arm.arm}</div>
                            <div className="flex-1 bg-gray-100 h-6 rounded-full overflow-hidden">
                              <div
                                className="h-full rounded-full transition-all"
                                style={{ width: `${barPct}%`, background: color }}
                              />
                            </div>
                            <span className="text-xs text-gray-600 font-mono w-20 text-right flex-shrink-0">{label}</span>
                            {isActive && <span className="text-[10px] font-bold flex-shrink-0" style={{ color }}>Grafik ↓</span>}
                          </button>
                        );
                      })}
                    </div>

                    {/* Stat table */}
                    <div className="mt-4 pt-3 border-t border-gray-100">
                      <table className="w-full text-xs">
                        <thead>
                          <tr className="text-gray-400 uppercase tracking-wide">
                            <th className="text-left font-semibold pb-1.5">Faz</th>
                            <th className="text-right font-semibold pb-1.5">Yeşil</th>
                            <th className="text-right font-semibold pb-1.5">Sarı</th>
                            <th className="text-right font-semibold pb-1.5">Kırmızı</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-50">
                          {phaseData.map((arm, i) => (
                            <tr
                              key={arm.arm}
                              onClick={() => setSelectedArm(selectedArm === arm.arm ? null : arm.arm)}
                              className={`cursor-pointer transition ${selectedArm === arm.arm ? 'bg-blue-50' : 'hover:bg-gray-50'
                                }`}
                            >
                              <td className="py-1 font-bold" style={{ color: ARM_PHASE_COLORS[i % ARM_PHASE_COLORS.length] }}>Faz {arm.arm}</td>
                              <td className="text-right text-green-600 font-mono">{arm.green}s</td>
                              <td className="text-right text-yellow-600 font-mono">{arm.yellow}s</td>
                              <td className="text-right text-red-500 font-mono">{arm.red}s</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>

                    {/* Color legend */}
                    <div className="flex gap-4 mt-3">
                      <span className="flex items-center gap-1.5 text-xs text-gray-400"><span className="h-2 w-5 rounded-sm bg-green-500" />Yeşil</span>
                      <span className="flex items-center gap-1.5 text-xs text-gray-400"><span className="h-2 w-5 rounded-sm bg-yellow-400" />Sarı</span>
                      <span className="flex items-center gap-1.5 text-xs text-gray-400"><span className="h-2 w-5 rounded-sm" style={{ background: '#334155' }} />Koruma</span>
                    </div>
                  </div>

                  <div className="hidden lg:block w-px bg-gray-100" />

                  {/* Right — Pie chart + legend */}
                  <div className="w-full lg:w-72 p-5 flex flex-col items-center border-t lg:border-t-0 border-gray-100">
                    <p className="text-[10px] font-bold uppercase tracking-widest text-gray-400 mb-1 self-start">
                      {phaseView === 'duration' ? 'Faz Süre Dağılımı' : 'Araç Sayısı Dağılımı'}
                    </p>
                    <ResponsiveContainer width="100%" height={220}>
                      <PieChart>
                        <Pie
                          data={phaseData.map((arm, i) => ({
                            name: `Faz ${arm.arm}`,
                            value: phaseView === 'duration' ? arm.green : Math.round(arm.vehicle_count),
                            fill: ARM_PHASE_COLORS[i % ARM_PHASE_COLORS.length],
                          }))}
                          cx="50%"
                          cy="50%"
                          innerRadius={52}
                          outerRadius={90}
                          paddingAngle={2}
                          dataKey="value"
                          label={({ name, value }: any) =>
                            `${name} ${value}${phaseView === 'duration' ? 's' : ''}`
                          }
                          labelLine
                        >
                          {phaseData.map((_, i) => (
                            <Cell key={i} fill={ARM_PHASE_COLORS[i % ARM_PHASE_COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip
                          formatter={(v: number) => [phaseView === 'duration' ? `${v}s` : `${v} araç`]}
                          contentStyle={{ fontSize: 11, borderRadius: 8 }}
                        />
                      </PieChart>
                    </ResponsiveContainer>

                    {/* Legend rows */}
                    <div className="space-y-2 w-full mt-1">
                      {phaseData.map((arm, i) => {
                        const total = phaseData.reduce((s, a) =>
                          s + (phaseView === 'duration' ? a.green : a.vehicle_count), 0);
                        const val = phaseView === 'duration' ? arm.green : arm.vehicle_count;
                        const pct = total > 0 ? Math.round((val / total) * 100) : 0;
                        return (
                          <div key={arm.arm} className="flex items-center gap-2 text-xs">
                            <div className="h-3 w-3 rounded-full flex-shrink-0" style={{ background: ARM_PHASE_COLORS[i % ARM_PHASE_COLORS.length] }} />
                            <span className="font-bold text-gray-700 w-10">Faz {arm.arm}</span>
                            <span className="text-gray-400 flex-1 truncate text-[11px]" title={arm.name}>{arm.name}</span>
                            <span className="font-bold text-gray-800 tabular-nums">
                              {phaseView === 'duration' ? `${arm.green}s` : Math.round(arm.vehicle_count)}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                </div>
              </div>
            )}
          </section>
        )}



        {/* ── Adım 4: Grafik ──────────────────────────────────────────── */}
        {selectedJunction && selectedArm && (
          <section className="animate-fade-in space-y-4">
            <div className="flex items-center justify-between flex-wrap gap-2">
              <div>
                <h2 className="text-base font-bold text-gray-800">
                  {selectedJunction.name} — Faz {selectedArm} — Araç Akışı
                </h2>
                <p className="text-xs text-gray-400 mt-0.5 flex items-center gap-1">
                  {source && <><Activity className="h-3 w-3" />{source === 'AFDGCN' ? 'Garnoldi' : source} · </>}
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
                      contentStyle={{ borderRadius: 10, border: '1px solid #e0e7ff', fontSize: 12, boxShadow: '0 4px 12px rgba(0,0,0,0.08)' }}
                      formatter={(v: number, name: string) => [`${v} araç`, name]}
                      labelStyle={{ fontWeight: 600, color: '#374151' }}
                    />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <Line
                      type="monotone" dataKey="real" name="Gerçek"
                      stroke="#2563eb" strokeWidth={2.5} dot={false} activeDot={{ r: 5, fill: '#2563eb' }} connectNulls
                    />
                    <Line
                      type="monotone" dataKey="predicted" name="Garnoldi Tahmini"
                      stroke="#f59e0b" strokeWidth={2.5} strokeDasharray="6 3"
                      dot={false} activeDot={{ r: 5, fill: '#f59e0b' }} connectNulls
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
