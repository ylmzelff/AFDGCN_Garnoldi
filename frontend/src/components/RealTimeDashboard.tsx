/**
 * Real-time Prediction Dashboard Component
 * ==========================================
 * 
 * Bölge seçili tahminleri ve kavşak detaylarını gösterir.
 * - Otomatik refresh (10 sn)
 * - Real-time vehicle count charts
 * - Phase recommendations
 */

import React, { useState, useEffect } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { RefreshCw, TrendingUp, AlertCircle } from 'lucide-react';
import * as api from '@/api/phases';
import { PredictionResponse, JunctionDetailResponse } from '@/api/types';

interface DashboardProps {
  initialRegion?: string;
}

export const RealTimeDashboard: React.FC<DashboardProps> = ({ initialRegion = 'ildem' }) => {
  const [region, setRegion] = useState(initialRegion);
  const [selectedJunction, setSelectedJunction] = useState<number | null>(null);
  const [refreshInterval, setRefreshInterval] = useState(10000); // 10 saniye
  const queryClient = useQueryClient();

  // Bölge tahmin sorgusu (otomatik refresh)
  const regionQuery = useQuery({
    queryKey: ['prediction', region],
    queryFn: () => api.predictRegion(region),
    refetchInterval: refreshInterval,
    staleTime: 5000,
  });

  // Kavşak detaylı tahmini
  const junctionQuery = useQuery({
    queryKey: ['prediction', region, selectedJunction],
    queryFn: selectedJunction ? () => api.predictJunction(selectedJunction, region) : () => Promise.resolve(null),
    enabled: !!selectedJunction,
    refetchInterval: selectedJunction ? refreshInterval : false,
  });

  // Bölge listesi
  const regionsQuery = useQuery({
    queryKey: ['regions'],
    queryFn: () => api.listRegions(),
    staleTime: 60000,
  });

  const regions = regionsQuery.data?.regions ?? [];
  const regionData = regionQuery.data as PredictionResponse | undefined;
  const junctionData = junctionQuery.data as JunctionDetailResponse | undefined;

  // Manual refresh
  const handleRefresh = () => {
    queryClient.invalidateQueries({ queryKey: ['prediction', region] });
    if (selectedJunction) {
      queryClient.invalidateQueries({ queryKey: ['prediction', region, selectedJunction] });
    }
  };

  if (regionQuery.isError) {
    return (
      <div className="p-6 bg-red-50 border border-red-200 rounded-lg flex items-center gap-3">
        <AlertCircle className="text-red-600" size={24} />
        <div>
          <h3 className="font-semibold text-red-900">Tahmin Hatası</h3>
          <p className="text-sm text-red-700">{(regionQuery.error as any)?.message || 'API bağlantısı kurulamadı'}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 p-6 bg-gradient-to-br from-slate-50 to-slate-100 min-h-screen">
      {/* Header */}
      <div className="flex items-center justify-between bg-white p-4 rounded-lg shadow">
        <div>
          <h1 className="text-3xl font-bold text-slate-900">🚦 Gerçek Zamanlı Tahminler</h1>
          <p className="text-sm text-slate-500">
            {regionData?.timestamp && new Date(regionData.timestamp).toLocaleTimeString('tr-TR')}
          </p>
        </div>
        <button
          onClick={handleRefresh}
          disabled={regionQuery.isLoading}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
        >
          <RefreshCw size={18} className={regionQuery.isLoading ? 'animate-spin' : ''} />
          Yenile
        </button>
      </div>

      {/* Region Selection & Refresh Interval */}
      <div className="bg-white p-4 rounded-lg shadow flex items-center gap-4">
        <div className="flex-1">
          <label className="block text-sm font-medium text-slate-700 mb-2">Bölge Seçin</label>
          <select
            value={region}
            onChange={(e) => {
              setRegion(e.target.value);
              setSelectedJunction(null);
            }}
            className="w-full px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            {regions.map((r: any) => (
              <option key={r.name} value={r.name}>
                {r.description} ({r.junction_count} kavşak)
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-2">Güncelleme Aralığı</label>
          <select
            value={refreshInterval}
            onChange={(e) => setRefreshInterval(Number(e.target.value))}
            className="px-3 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value={5000}>5 saniye</option>
            <option value={10000}>10 saniye</option>
            <option value={30000}>30 saniye</option>
            <option value={60000}>1 dakika</option>
          </select>
        </div>
      </div>

      {/* Status Info */}
      {regionData && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white p-4 rounded-lg shadow">
            <p className="text-sm text-slate-600">Veri Kaynağı</p>
            <p className="text-lg font-semibold text-slate-900">
              {regionData.source === 'AFDGCN' ? '🤖 Garnoldi' : '📊 Ortalama'}
            </p>
          </div>
          <div className="bg-white p-4 rounded-lg shadow">
            <p className="text-sm text-slate-600">API Durumu</p>
            <p className={`text-lg font-semibold ${regionData.kayseri_ok ? 'text-green-600' : 'text-red-600'}`}>
              {regionData.kayseri_ok ? '✅ Bağlı' : '❌ Kopuk'}
            </p>
          </div>
          <div className="bg-white p-4 rounded-lg shadow">
            <p className="text-sm text-slate-600">Kavşak Sayısı</p>
            <p className="text-lg font-semibold text-slate-900">{Object.keys(regionData.predictions).length}</p>
          </div>
          <div className="bg-white p-4 rounded-lg shadow">
            <p className="text-sm text-slate-600">Toplam Araç</p>
            <p className="text-lg font-semibold text-slate-900">
              {Object.values(regionData.predictions)
                .flatMap((arms) => Object.values(arms))
                .reduce((a, b) => a + b, 0)
                .toFixed(0)}
            </p>
          </div>
        </div>
      )}

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Junction List */}
        <div className="lg:col-span-1 bg-white rounded-lg shadow overflow-hidden">
          <div className="p-4 border-b bg-slate-50">
            <h2 className="font-semibold text-slate-900">Kavşaklar</h2>
          </div>
          <div className="divide-y max-h-96 overflow-y-auto">
            {regionData?.predictions &&
              Object.entries(regionData.predictions).map(([jid, arms]) => {
                const totalVehicles = Object.values(arms).reduce((a, b) => a + b, 0);
                const phaseData = regionData.phases[parseInt(jid)];
                const junctionName = Object.keys(arms).length > 0 ? `Kavşak ${jid}` : `ID: ${jid}`;

                return (
                  <button
                    key={jid}
                    onClick={() => setSelectedJunction(parseInt(jid))}
                    className={`w-full text-left p-4 hover:bg-blue-50 transition ${
                      selectedJunction === parseInt(jid) ? 'bg-blue-100 border-l-4 border-blue-600' : ''
                    }`}
                  >
                    <h3 className="font-medium text-slate-900">{junctionName}</h3>
                    <p className="text-sm text-slate-600">
                      🚗 {totalVehicles.toFixed(1)} araç • ⏱️ {phaseData?._cycle_time ?? 60}s
                    </p>
                  </button>
                );
              })}
          </div>
        </div>

        {/* Right: Details & Charts */}
        <div className="lg:col-span-2 space-y-6">
          {selectedJunction && junctionData ? (
            <>
              {/* Junction Header */}
              <div className="bg-white p-6 rounded-lg shadow">
                <h2 className="text-2xl font-bold text-slate-900">{junctionData.junction_name}</h2>
                <div className="grid grid-cols-2 gap-4 mt-4">
                  <div>
                    <p className="text-sm text-slate-600">Toplam Araç</p>
                    <p className="text-2xl font-bold text-blue-600">{junctionData.phase_recommendation.total_vehicles}</p>
                  </div>
                  <div>
                    <p className="text-sm text-slate-600">Çevrim Süresi</p>
                    <p className="text-2xl font-bold text-green-600">{junctionData.phase_recommendation.cycle_time}s</p>
                  </div>
                </div>
              </div>

              {/* Arms */}
              <div className="bg-white p-6 rounded-lg shadow">
                <h3 className="text-lg font-semibold text-slate-900 mb-4">Yol Kolları</h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  {Object.entries(junctionData.arms).map(([arm, data]: [string, any]) => (
                    <div key={arm} className="p-4 border border-slate-200 rounded-lg">
                      <div className="flex justify-between items-center mb-3">
                        <h4 className="font-semibold text-slate-900">Kol {arm}</h4>
                        <span className={`px-2 py-1 rounded text-xs font-semibold ${
                          data.status === 'high' ? 'bg-red-100 text-red-800' :
                          data.status === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                          'bg-green-100 text-green-800'
                        }`}>
                          {data.status}
                        </span>
                      </div>
                      <p className="text-sm text-slate-600 mb-2">{data.name}</p>
                      <div className="space-y-1 text-sm">
                        <p>🚗 Araçlar: <span className="font-semibold">{data.vehicle_count.toFixed(1)}</span></p>
                        <p>🛣️ Şerit: <span className="font-semibold">{data.lanes}</span></p>
                        <p>📊 Yük: <span className="font-semibold">{(data.load * 100).toFixed(0)}%</span></p>
                      </div>
                      <div className="mt-3 flex gap-1 text-xs">
                        <span className="px-2 py-1 bg-green-100 text-green-800 rounded">🟢 Y:{data.green}s</span>
                        <span className="px-2 py-1 bg-yellow-100 text-yellow-800 rounded">🟡 S:{data.yellow}s</span>
                        <span className="px-2 py-1 bg-red-100 text-red-800 rounded">🔴 K:{data.red}s</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          ) : (
            <div className="bg-slate-100 p-6 rounded-lg text-center text-slate-600">
              <TrendingUp className="mx-auto mb-2 text-slate-400" size={32} />
              <p>Detaylı görmek için sol taraftan bir kavşak seçin</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default RealTimeDashboard;
