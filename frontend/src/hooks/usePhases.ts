import { useQuery } from '@tanstack/react-query';
import { getIldemPhases, getTunaPhases } from '@/api/phases';
import { usePhaseStore } from '@/store/usePhaseStore';
import { useEffect } from 'react';

export function useIldemPhases() {
  const setIldem = usePhaseStore((s) => s.setIldem);

  const query = useQuery({
    queryKey: ['phases', 'ildem'],
    queryFn: getIldemPhases,
    staleTime: 60_000,
    refetchInterval: 600_000,
    retry: 2,
  });

  useEffect(() => {
    if (query.data) setIldem(query.data);
  }, [query.data, setIldem]);

  return query;
}

export function useTunaPhases() {
  const setTuna = usePhaseStore((s) => s.setTuna);

  const query = useQuery({
    queryKey: ['phases', 'tuna'],
    queryFn: getTunaPhases,
    staleTime: 60_000,
    refetchInterval: 600_000,
    retry: 2,
  });

  useEffect(() => {
    if (query.data) setTuna(query.data);
  }, [query.data, setTuna]);

  return query;
}
