import { Car } from 'lucide-react';
import type { ArmPhase } from '@/api/types';
import { StatusBadge } from './StatusBadge';
import { PhaseBar } from './PhaseBar';

interface Props {
  arm: ArmPhase;
}

export function ArmCard({ arm }: Props) {
  return (
    <div className="rounded-lg border border-gray-100 bg-white p-3 shadow-sm transition hover:shadow-md animate-fade-in">
      {/* Başlık */}
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="flex h-6 w-6 items-center justify-center rounded-full bg-brand text-xs font-bold text-white">
            {arm.arm}
          </span>
          <span className="text-sm font-semibold text-gray-800 truncate max-w-[120px]" title={arm.arm_name}>
            {arm.arm_name}
          </span>
        </div>
        <StatusBadge status={arm.status} />
      </div>

      {/* Metrikler */}
      <div className="mb-3 grid grid-cols-3 gap-1 text-center">
        <Metric label="Araç" value={arm.vehicle_count} icon={<Car className="h-3 w-3" />} />
        <Metric label="Şerit" value={arm.lanes} />
        <Metric label="Yük" value={arm.load.toFixed(1)} />
      </div>

      {/* Faz Bar */}
      <PhaseBar
        green={arm.green}
        yellow={arm.yellow}
        red={arm.red}
        cycle={arm.cycle_time}
      />
    </div>
  );
}

function Metric({
  label,
  value,
  icon,
}: {
  label: string;
  value: string | number;
  icon?: React.ReactNode;
}) {
  return (
    <div className="rounded bg-gray-50 p-1">
      <div className="flex items-center justify-center gap-0.5 text-gray-400 text-[10px]">
        {icon}
        {label}
      </div>
      <div className="text-center text-sm font-bold text-gray-700">{value}</div>
    </div>
  );
}
