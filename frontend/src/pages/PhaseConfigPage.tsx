import { useState, useEffect, useCallback, useRef } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { ArrowLeft, Settings, Save, RotateCcw, TrafficCone } from 'lucide-react';
import { apiClient } from '@/api/client';
import { junctionName } from '@/config/junctionMeta';
import { PhaseConfigFields, type PhaseConfigItem } from '@/components/JunctionPhaseConfig';

interface RegionOption {
  name: string;
  city: string;
}

function JunctionConfigCard({ item, city, onSaved }: { item: PhaseConfigItem; city: string; onSaved: (item: PhaseConfigItem) => void }) {
  const [draft, setDraft] = useState<PhaseConfigItem>(item);
  const [saving, setSaving] = useState(false);

  useEffect(() => { setDraft(item); }, [item]);

  const save = async () => {
    setSaving(true);
    try {
      const res = await apiClient.put('/admin/phase-configs', draft);
      onSaved(res.data);
    } finally {
      setSaving(false);
    }
  };

  const reset = async () => {
    setSaving(true);
    try {
      await apiClient.delete(`/admin/phase-configs/${draft.region}/${draft.junctionId}`);
      const res = await apiClient.get(`/admin/phase-configs?region=${draft.region}`);
      const fresh = (res.data as PhaseConfigItem[]).find((x) => x.junctionId === draft.junctionId);
      if (fresh) onSaved(fresh);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="rounded-2xl border border-gray-100 bg-white p-5 shadow-sm">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <TrafficCone className="h-4 w-4 text-brand" />
          <span className="font-bold text-gray-800 text-sm">{junctionName(city, draft.junctionId)}</span>
          <span className="text-[11px] text-gray-400">#{draft.junctionId}</span>
        </div>
        <span className={`text-[10px] font-bold uppercase tracking-wide px-2 py-0.5 rounded-full ${draft.isCustom ? 'bg-brand/10 text-brand' : 'bg-gray-100 text-gray-400'}`}>
          {draft.isCustom ? 'Özel' : 'Varsayılan'}
        </span>
      </div>

      <div className="mb-4">
        <PhaseConfigFields draft={draft} setDraft={setDraft} />
      </div>

      <div className="flex items-center gap-2 pt-2 border-t border-gray-50">
        <button
          onClick={save}
          disabled={saving}
          className="flex items-center gap-1.5 rounded-lg bg-brand text-white text-xs font-semibold px-3 py-1.5 hover:bg-brand/90 transition disabled:opacity-50"
        >
          <Save className="h-3.5 w-3.5" />Kaydet
        </button>
        {draft.isCustom && (
          <button
            onClick={reset}
            disabled={saving}
            className="flex items-center gap-1.5 rounded-lg border border-gray-200 text-gray-500 text-xs font-semibold px-3 py-1.5 hover:bg-gray-50 transition disabled:opacity-50"
          >
            <RotateCcw className="h-3.5 w-3.5" />Varsayılana döndür
          </button>
        )}
      </div>
    </div>
  );
}

export default function PhaseConfigPage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const targetJunctionId = searchParams.get('junction') ? Number(searchParams.get('junction')) : null;
  const targetCardRef = useRef<HTMLDivElement | null>(null);

  const [regions, setRegions] = useState<RegionOption[]>([]);
  const [selectedRegion, setSelectedRegion] = useState<string | null>(null);
  const [items, setItems] = useState<PhaseConfigItem[] | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    apiClient.get('/predict/regions').then((res) => {
      type ApiRegion = { name: string; city: string };
      const data: ApiRegion[] = res.data?.regions ?? [];
      setRegions(data.map((r) => ({ name: r.name, city: r.city })));
      const fromUrl = searchParams.get('region');
      setSelectedRegion(fromUrl && data.some((r) => r.name === fromUrl) ? fromUrl : (data[0]?.name ?? null));
    }).catch(() => { /* sessizce geç */ });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (items && targetJunctionId != null && targetCardRef.current) {
      targetCardRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [items, targetJunctionId]);

  const loadItems = useCallback(async (region: string) => {
    setLoading(true);
    try {
      const res = await apiClient.get(`/admin/phase-configs?region=${region}`);
      setItems(res.data);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (selectedRegion) loadItems(selectedRegion);
  }, [selectedRegion, loadItems]);

  const currentCity = regions.find((r) => r.name === selectedRegion)?.city ?? '';

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="sticky top-0 z-20 bg-brand shadow-lg">
        <div className="flex w-full items-center justify-between px-6 py-3">
          <div className="flex items-center gap-3">
            <button
              onClick={() => navigate('/dashboard')}
              className="flex items-center gap-1.5 rounded-lg px-2 py-1.5 text-sm text-white/80 hover:bg-white/10 hover:text-white transition"
            >
              <ArrowLeft className="h-4 w-4" />
            </button>
            <Settings className="h-6 w-6 text-white" />
            <div>
              <h1 className="text-base font-bold text-white leading-tight">Faz Ayarları</h1>
              <p className="text-xs text-white/50 leading-tight">Kavşak bazında sarı süre, min yeşil, şerit sayısı</p>
            </div>
          </div>
        </div>
      </header>

      <main className="w-full px-6 py-6 space-y-6">
        {/* Bölge seçimi */}
        <div className="flex flex-wrap gap-2">
          {regions.map((r) => (
            <button
              key={r.name}
              onClick={() => setSelectedRegion(r.name)}
              className={`rounded-full border px-3.5 py-1.5 text-sm font-medium transition ${selectedRegion === r.name
                  ? 'border-brand bg-brand text-white shadow-md'
                  : 'border-gray-200 bg-white text-gray-700 hover:border-brand/50'
                }`}
            >
              {r.city.toUpperCase()} · {r.name.toUpperCase()}
            </button>
          ))}
        </div>

        {loading && <p className="text-sm text-gray-400">Yükleniyor...</p>}

        {!loading && items && (
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
            {items.map((item) => {
              const isTarget = item.junctionId === targetJunctionId;
              return (
                <div
                  key={item.junctionId}
                  ref={isTarget ? targetCardRef : undefined}
                  className={isTarget ? 'rounded-2xl ring-2 ring-brand ring-offset-2' : ''}
                >
                  <JunctionConfigCard
                    item={item}
                    city={currentCity}
                    onSaved={(saved) => setItems((prev) => prev?.map((x) => (x.junctionId === saved.junctionId ? saved : x)) ?? null)}
                  />
                </div>
              );
            })}
          </div>
        )}
      </main>
    </div>
  );
}
