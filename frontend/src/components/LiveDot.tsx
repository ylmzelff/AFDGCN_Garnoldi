import clsx from 'clsx';

type WsStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

interface Props {
  status: WsStatus;
}

const CONFIG: Record<WsStatus, { label: string; dotCls: string }> = {
  connected:    { label: 'Canlı',       dotCls: 'bg-green-500 animate-pulse-slow' },
  connecting:   { label: 'Bağlanıyor…', dotCls: 'bg-yellow-400 animate-pulse' },
  disconnected: { label: 'Bağlantı Yok', dotCls: 'bg-gray-400' },
  error:        { label: 'Hata',         dotCls: 'bg-red-500 animate-pulse' },
};

export function LiveDot({ status }: Props) {
  const { label, dotCls } = CONFIG[status];
  return (
    <span className="inline-flex items-center gap-1.5 text-sm font-medium text-gray-200">
      <span className={clsx('inline-block h-2 w-2 rounded-full', dotCls)} />
      {label}
    </span>
  );
}
