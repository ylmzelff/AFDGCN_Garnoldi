import { useState, useEffect } from 'react';
import { Save, RotateCcw, Settings } from 'lucide-react';
import { apiClient } from '@/api/client';
import { junctionName } from '@/config/junctionMeta';
import { PhaseConfigFields, type PhaseConfigItem } from './JunctionPhaseConfig';

interface Props {
  region: string;
  city: string;
  junctionId: number;
  /** Kaydet başarıyla tamamlanınca çağrılır — üst bileşen faz önerisini tetikler. */
  onSaved: () => void;
}

/**
 * Kavşak seçilince doğrudan görünen faz ayarları bölümü (modal değil).
 * "Kaydet"e basınca üst bileşen faz önerisini otomatik çeker —
 * ayrı bir "Faz Önerisi Al" butonuna gerek kalmaz.
 */
export function JunctionPhaseInline({ region, city, junctionId, onSaved }: Props) {
  const [draft, setDraft] = useState<PhaseConfigItem | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setDraft(null);
    apiClient.get(`/admin/phase-configs?region=${region}`).then((res) => {
      if (cancelled) return;
      const item = (res.data as PhaseConfigItem[]).find((x) => x.junctionId === junctionId) ?? null;
      setDraft(item);
    }).finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [region, junctionId]);

  const save = async () => {
    if (!draft) return;
    setSaving(true);
    try {
      await apiClient.put('/admin/phase-configs', draft);
      onSaved();
    } finally {
      setSaving(false);
    }
  };

  const reset = async () => {
    setSaving(true);
    try {
      await apiClient.delete(`/admin/phase-configs/${region}/${junctionId}`);
      const res = await apiClient.get(`/admin/phase-configs?region=${region}`);
      const fresh = (res.data as PhaseConfigItem[]).find((x) => x.junctionId === junctionId);
      if (fresh) setDraft(fresh);
    } finally {
      setSaving(false);
    }
  };

  return (
    <section className="animate-fade-in">
      <div className="rounded-2xl border border-gray-100 bg-white shadow-sm overflow-hidden">
        <div className="flex items-center justify-between px-5 py-3.5 bg-gray-50 border-b border-gray-100">
          <div className="flex items-center gap-2 text-gray-800">
            <Settings className="h-4 w-4 text-brand" />
            <span className="font-bold text-sm">{junctionName(city, junctionId)} Faz Ayarları</span>
          </div>
          {draft && (
            <span className={`text-[10px] font-bold uppercase tracking-wide px-2 py-0.5 rounded-full ${draft.isCustom ? 'bg-brand/10 text-brand' : 'bg-gray-100 text-gray-400'}`}>
              {draft.isCustom ? 'Özel' : 'Varsayılan'}
            </span>
          )}
        </div>

        <div className="p-5">
          {loading && <p className="text-sm text-gray-400">Yükleniyor...</p>}
          {!loading && !draft && <p className="text-sm text-gray-400">Ayar bulunamadı.</p>}
          {!loading && draft && (
            <PhaseConfigFields draft={draft} setDraft={(updater) => setDraft((d) => (d ? updater(d) : d))} />
          )}
        </div>

        {!loading && draft && (
          <div className="flex items-center gap-2 px-5 py-3.5 border-t border-gray-100">
            <button
              onClick={save}
              disabled={saving}
              className="flex items-center gap-1.5 rounded-lg bg-brand text-white text-xs font-semibold px-4 py-2 hover:bg-brand/90 transition disabled:opacity-50"
            >
              <Save className="h-3.5 w-3.5" />{saving ? 'Kaydediliyor…' : 'Kaydet ve Faz Önerisini Göster'}
            </button>
            {draft.isCustom && (
              <button
                onClick={reset}
                disabled={saving}
                className="flex items-center gap-1.5 rounded-lg border border-gray-200 text-gray-500 text-xs font-semibold px-3 py-2 hover:bg-gray-50 transition disabled:opacity-50"
              >
                <RotateCcw className="h-3.5 w-3.5" />Varsayılana döndür
              </button>
            )}
          </div>
        )}
      </div>
    </section>
  );
}
