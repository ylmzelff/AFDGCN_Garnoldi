import type { ArmPhase } from '@/api/types';

interface Props {
  green: number;
  yellow: number;
  red: number;
  cycle: number;
  /** Animasyonlu trafik lambası mı gösterilsin? */
  showLight?: boolean;
  arm?: ArmPhase;
}

/**
 * Yatay stacked bar: yeşil | sarı | kırmızı
 * Oranlar cycle_time'a göre hesaplanır.
 */
export function PhaseBar({ green, yellow, red, cycle, showLight, arm }: Props) {
  const pct = (v: number) => `${Math.round((v / cycle) * 100)}%`;

  return (
    <div className="space-y-1">
      {/* Bar */}
      <div className="flex h-5 w-full overflow-hidden rounded-full bg-gray-200">
        <div
          className="flex items-center justify-center bg-green-500 text-[10px] font-bold text-white transition-all duration-700"
          style={{ width: pct(green) }}
          title={`Yeşil: ${green}s`}
        >
          {green > 8 ? `${green}s` : ''}
        </div>
        <div
          className="flex items-center justify-center bg-yellow-400 text-[10px] font-bold text-white transition-all duration-700"
          style={{ width: pct(yellow) }}
          title={`Sarı: ${yellow}s`}
        />
        <div
          className="flex items-center justify-center bg-red-500 text-[10px] font-bold text-white transition-all duration-700"
          style={{ width: pct(red) }}
          title={`Kırmızı: ${red}s`}
        >
          {red > 8 ? `${red}s` : ''}
        </div>
      </div>

      {/* Etiketler */}
      <div className="flex justify-between text-[10px] text-gray-400">
        <span className="text-green-600 font-semibold">{green}s yeşil</span>
        <span className="text-yellow-500">{yellow}s</span>
        <span className="text-red-500 font-semibold">{red}s kırmızı</span>
      </div>

      {/* Trafik lambası ikonları */}
      {showLight && arm && (
        <div className="flex gap-1.5 mt-1">
          <TrafficLightIcon active="green" value={arm.green} />
          <TrafficLightIcon active="yellow" value={arm.yellow} />
          <TrafficLightIcon active="red" value={arm.red} />
        </div>
      )}
    </div>
  );
}

function TrafficLightIcon({
  active,
  value,
}: {
  active: 'green' | 'yellow' | 'red';
  value: number;
}) {
  const colors = {
    green:  'bg-green-500',
    yellow: 'bg-yellow-400',
    red:    'bg-red-500',
  };
  return (
    <div className="flex flex-col items-center gap-0.5 rounded bg-gray-800 px-1.5 py-1">
      <div className={`h-2.5 w-2.5 rounded-full ${colors[active]}`} />
      <span className="text-[9px] text-gray-300">{value}s</span>
    </div>
  );
}
