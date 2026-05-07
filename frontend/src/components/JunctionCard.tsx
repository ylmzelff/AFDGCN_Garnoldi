import { ChevronDown, ChevronUp, TrafficCone } from 'lucide-react';
import { useState } from 'react';
import type { JunctionPhase } from '@/api/types';
import { ArmCard } from './ArmCard';
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from 'recharts';

const PIE_COLORS = ['#3b82f6', '#22c55e', '#eab308', '#ef4444', '#8b5cf6', '#06b6d4'];

interface Props {
  junction: JunctionPhase;
}

export function JunctionCard({ junction }: Props) {
  const [expanded, setExpanded] = useState(true);

  const pieData = junction.arms.map((a) => ({
    name: `${a.arm} – ${a.arm_name}`,
    value: a.vehicle_count,
  }));

  return (
    <div className="rounded-xl border border-gray-200 bg-white shadow-sm overflow-hidden animate-fade-in">
      {/* Header */}
      <button
        className="flex w-full items-center justify-between px-4 py-3 bg-gradient-to-r from-brand to-brand-light text-white hover:opacity-95 transition"
        onClick={() => setExpanded((v) => !v)}
      >
        <div className="flex items-center gap-2">
          <TrafficCone className="h-4 w-4 opacity-80" />
          <span className="font-bold text-sm">{junction.junction_name}</span>
          <span className="text-xs opacity-70">#{junction.junction_id}</span>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs opacity-80">
            {junction.total_vehicles} araç · {junction.cycle_time}s
          </span>
          {expanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
        </div>
      </button>

      {expanded && (
        <div className="p-4">
          <div className="flex flex-col gap-4 lg:flex-row">
            {/* Kol Kartları */}
            <div className="grid flex-1 grid-cols-1 gap-2 sm:grid-cols-2">
              {junction.arms.map((arm) => (
                <ArmCard key={arm.arm} arm={arm} />
              ))}
            </div>

            {/* Pasta Grafik */}
            {junction.total_vehicles > 0 && (
              <div className="flex flex-col items-center justify-center w-full lg:w-48">
                <span className="mb-1 text-xs font-semibold text-gray-400 uppercase tracking-wide">
                  Araç Dağılımı
                </span>
                <ResponsiveContainer width={160} height={160}>
                  <PieChart>
                    <Pie
                      data={pieData}
                      cx="50%"
                      cy="50%"
                      innerRadius={40}
                      outerRadius={72}
                      paddingAngle={3}
                      dataKey="value"
                    >
                      {pieData.map((_, i) => (
                        <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip
                      formatter={(v: number) => [`${v} araç`, '']}
                      contentStyle={{ fontSize: 11, borderRadius: 8 }}
                    />
                  </PieChart>
                </ResponsiveContainer>
                {/* Lejand */}
                <div className="mt-1 space-y-0.5 w-full">
                  {pieData.map((d, i) => (
                    <div key={i} className="flex items-center gap-1.5 text-[10px] text-gray-600">
                      <div
                        className="h-2 w-2 rounded-full flex-shrink-0"
                        style={{ backgroundColor: PIE_COLORS[i % PIE_COLORS.length] }}
                      />
                      <span className="truncate">{d.name}</span>
                      <span className="ml-auto font-semibold">{d.value}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
