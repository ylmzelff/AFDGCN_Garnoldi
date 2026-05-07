import { BarChart2, Brain, AlertTriangle, RefreshCw } from 'lucide-react';
import type { RegionPhaseResponse } from '@/api/types';
import { JunctionCard } from './JunctionCard';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';

const SOURCE_COLORS: Record<string, string> = {
  AFDGCN:         '#3b82f6',
  moving_average: '#8b5cf6',
};

interface Props {
  data: RegionPhaseResponse;
  isLoading?: boolean;
  isError?: boolean;
  onRefresh?: () => void;
}

export function RegionPanel({ data, isLoading, isError, onRefresh }: Props) {
  const totalVehicles = data.junctions.reduce((s, j) => s + j.total_vehicles, 0);

  const barData = data.junctions.map((j) => ({
    name: j.junction_name,
    vehicles: j.total_vehicles,
  }));

  return (
    <div className="space-y-4 animate-fade-in">
      {/* Bölge Özeti */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <SummaryCard
          label="Toplam Araç"
          value={totalVehicles}
          icon={<BarChart2 className="h-4 w-4 text-blue-500" />}
        />
        <SummaryCard
          label="Kavşak Sayısı"
          value={data.junctions.length}
          icon={<BarChart2 className="h-4 w-4 text-purple-500" />}
        />
        <SummaryCard
          label="Tahmin Kaynağı"
          value={data.prediction_source === 'AFDGCN' ? 'AFDGCN' : 'Avg'}
          icon={<Brain className="h-4 w-4" style={{ color: SOURCE_COLORS[data.prediction_source] }} />}
          highlight={data.prediction_source === 'AFDGCN' ? 'blue' : 'purple'}
        />
        <SummaryCard
          label="Belediye API"
          value={data.kayseri_api_status === 'connected' ? 'Bağlı' : 'Yok'}
          icon={
            data.kayseri_api_status === 'connected'
              ? <div className="h-2 w-2 rounded-full bg-green-500" />
              : <AlertTriangle className="h-4 w-4 text-yellow-500" />
          }
          highlight={data.kayseri_api_status === 'connected' ? 'green' : 'yellow'}
        />
      </div>

      {/* Hata/Yükleniyor Banner */}
      {isLoading && (
        <div className="flex items-center gap-2 rounded-lg bg-blue-50 px-4 py-2 text-sm text-blue-700">
          <RefreshCw className="h-4 w-4 animate-spin" />
          Veri yükleniyor…
        </div>
      )}
      {isError && (
        <div className="flex items-center justify-between rounded-lg bg-red-50 px-4 py-2 text-sm text-red-700">
          <span>Veri alınamadı.</span>
          {onRefresh && (
            <button
              onClick={onRefresh}
              className="flex items-center gap-1 rounded bg-red-100 px-2 py-1 text-xs font-semibold hover:bg-red-200 transition"
            >
              <RefreshCw className="h-3 w-3" /> Tekrar Dene
            </button>
          )}
        </div>
      )}

      {/* Kavşak Araç Sayısı Bar Grafiği */}
      <div className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm">
        <h3 className="mb-3 text-sm font-semibold text-gray-600 uppercase tracking-wide">
          Kavşak Bazlı Araç Yoğunluğu
        </h3>
        <ResponsiveContainer width="100%" height={180}>
          <BarChart data={barData} margin={{ top: 4, right: 8, left: -10, bottom: 40 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis
              dataKey="name"
              tick={{ fontSize: 10 }}
              angle={-30}
              textAnchor="end"
              interval={0}
            />
            <YAxis tick={{ fontSize: 10 }} />
            <Tooltip
              formatter={(v: number) => [`${v} araç`, 'Araç Sayısı']}
              contentStyle={{ fontSize: 11, borderRadius: 8 }}
            />
            <Bar dataKey="vehicles" radius={[4, 4, 0, 0]}>
              {barData.map((_, i) => (
                <Cell key={i} fill={i % 2 === 0 ? '#3b82f6' : '#60a5fa'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Son güncelleme */}
      <p className="text-right text-xs text-gray-400">
        Zaman dilimi: <span className="font-semibold text-gray-600">{data.time_label}</span>
        {' · '}
        {new Date(data.timestamp).toLocaleTimeString('tr-TR')}
      </p>

      {/* Kavşak Kartları */}
      <div className="space-y-3">
        {data.junctions.map((j) => (
          <JunctionCard key={j.junction_id} junction={j} />
        ))}
      </div>
    </div>
  );
}

function SummaryCard({
  label,
  value,
  icon,
  highlight,
}: {
  label: string;
  value: string | number;
  icon?: React.ReactNode;
  highlight?: 'blue' | 'green' | 'yellow' | 'purple';
}) {
  const ringColor = {
    blue:   'ring-blue-200',
    green:  'ring-green-200',
    yellow: 'ring-yellow-200',
    purple: 'ring-purple-200',
  }[highlight ?? 'blue'] ?? '';

  return (
    <div className={`rounded-xl border bg-white p-3 shadow-sm ring-1 ${ringColor}`}>
      <div className="flex items-center gap-1.5 text-xs text-gray-400 mb-1">
        {icon}
        {label}
      </div>
      <div className="text-lg font-bold text-gray-800">{value}</div>
    </div>
  );
}
