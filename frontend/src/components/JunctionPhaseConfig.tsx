export interface PhaseConfigItem {
  city: string;
  region: string;
  junctionId: number;
  lanes: Record<string, number>;
  fixedYellow: number;
  fixedProtection: number;
  minGreen: number;
  cycleMin: number;
  cycleMax: number;
  threshLow: number;
  threshHigh: number;
  isCustom: boolean;
}

export function NumberField({ label, value, onChange }: { label: string; value: number; onChange: (v: number) => void }) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-[11px] font-medium text-gray-500">{label}</span>
      <input
        type="number"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full rounded-lg border border-gray-200 px-2.5 py-1.5 text-sm focus:border-brand focus:outline-none focus:ring-1 focus:ring-brand"
      />
    </label>
  );
}

/** Kavşak faz ayarları form alanları — şerit sayısı + süre ayarları. */
export function PhaseConfigFields({ draft, setDraft }: { draft: PhaseConfigItem; setDraft: (updater: (d: PhaseConfigItem) => PhaseConfigItem) => void }) {
  const setLane = (arm: string, v: number) => setDraft((d) => ({ ...d, lanes: { ...d.lanes, [arm]: v } }));

  return (
    <>
      <p className="text-[10px] font-bold uppercase tracking-widest text-gray-400 mb-2">Şerit Sayısı</p>
      <div className="grid grid-cols-4 gap-2 mb-4">
        {Object.keys(draft.lanes).map((arm) => (
          <NumberField key={arm} label={`Kol ${arm}`} value={draft.lanes[arm]} onChange={(v) => setLane(arm, v)} />
        ))}
      </div>

      <p className="text-[10px] font-bold uppercase tracking-widest text-gray-400 mb-2">Süre Ayarları (saniye)</p>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
        <NumberField label="Sarı süre" value={draft.fixedYellow} onChange={(v) => setDraft((d) => ({ ...d, fixedYellow: v }))} />
        <NumberField label="Koruma süresi" value={draft.fixedProtection} onChange={(v) => setDraft((d) => ({ ...d, fixedProtection: v }))} />
        <NumberField label="Min yeşil" value={draft.minGreen} onChange={(v) => setDraft((d) => ({ ...d, minGreen: v }))} />
        <NumberField label="Min döngü" value={draft.cycleMin} onChange={(v) => setDraft((d) => ({ ...d, cycleMin: v }))} />
        <NumberField label="Max döngü" value={draft.cycleMax} onChange={(v) => setDraft((d) => ({ ...d, cycleMax: v }))} />
      </div>
    </>
  );
}
