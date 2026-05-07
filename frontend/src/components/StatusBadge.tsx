import clsx from 'clsx';
import type { ArmStatus } from '@/api/types';

interface Props {
  status: ArmStatus;
  className?: string;
}

const CONFIG: Record<ArmStatus, { label: string; cls: string }> = {
  low:    { label: 'Düşük',   cls: 'bg-green-100 text-green-800 ring-green-200' },
  medium: { label: 'Orta',    cls: 'bg-yellow-100 text-yellow-800 ring-yellow-200' },
  high:   { label: 'Yüksek', cls: 'bg-red-100 text-red-800 ring-red-200' },
};

export function StatusBadge({ status, className }: Props) {
  const { label, cls } = CONFIG[status];
  return (
    <span
      className={clsx(
        'inline-flex items-center rounded-full px-2 py-0.5 text-xs font-semibold ring-1 ring-inset',
        cls,
        className
      )}
    >
      {label}
    </span>
  );
}
