import { CheckCircle2, Clock, Film, Layers, Sparkles } from 'lucide-react';
import { useEffect, useState } from 'react';
import type { Session, SessionSnapshot } from '../app/session';
import type { LoadedModel } from '../ml/model-client';
import type { Segment } from '../ml/segmenter';
// import type { Decision } from '../ml/decision';
// import { GuessList } from './GuessList';
// import { Percent } from './ui';

interface SegmentHistoryEntry {
  id: number;
  segment: Segment;
  partial: boolean;
  timeStr: string;
}

const HISTORY = 8;

export function RecognisePanel({
  session,
  snap,
}: {
  session: Session;
  snap: SessionSnapshot;
  model: LoadedModel;
}) {
  const [history, setHistory] = useState<SegmentHistoryEntry[]>([]);

  useEffect(
    () =>
      session.onResult((result, id) => {
        const timeStr = new Date().toLocaleTimeString();
        setHistory((h) => [{ id, segment: result.segment, partial: result.partial, timeStr }, ...h].slice(0, HISTORY));
      }),
    [session],
  );

  const latest = history[0];

  return (
    <div className="space-y-5">
      <div>
        <span className="inline-flex items-center gap-1.5 rounded-md bg-accent-soft px-2 py-0.5 text-xs font-semibold text-accent">
          <Sparkles className="size-3.5" /> Báo cáo tiến độ: Bước 3
        </span>
        <h2 className="mt-1 text-lg font-bold">Phân đoạn cử chỉ tay (Sign Segmentation)</h2>
        <p className="mt-1 text-sm text-ink-muted leading-relaxed">
          Thực hiện cử chỉ trước webcam. Khi bạn <strong>dừng tay lại 300ms</strong> hoặc <strong>hạ tay xuống</strong>, thuật toán phân đoạn sẽ tự động bắt trọn vẹn chuỗi khung hình của cử chỉ.
        </p>
      </div>

      <div
        className="rounded-xl border border-line bg-paper p-4"
        aria-live="polite"
        data-testid="recognise-result"
      >
        {snap.signing ? (
          <div className="flex items-center gap-3 py-2 text-accent">
            <span aria-hidden className="pulse-dot size-3 rounded-full bg-accent" />
            <div>
              <p className="text-base font-semibold">Đang phát hiện cử chỉ tay...</p>
              <p className="text-xs text-ink-muted">Thuật toán Segmenter đang theo dõi và gom các frames chuyển động</p>
            </div>
          </div>
        ) : latest ? (
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-emerald-600 font-semibold">
              <CheckCircle2 className="size-5" />
              <span>Đã phân đoạn thành công 1 cử chỉ!</span>
            </div>

            <div className="grid grid-cols-3 gap-2 pt-1 text-center">
              <div className="rounded-lg border border-line bg-surface p-2.5">
                <div className="flex justify-center items-center gap-1 text-xs text-ink-muted">
                  <Film className="size-3.5" /> Khung hình
                </div>
                <div className="mt-1 text-xl font-bold">{latest.segment.frames.length}</div>
                <div className="text-[10px] text-ink-muted">frames</div>
              </div>

              <div className="rounded-lg border border-line bg-surface p-2.5">
                <div className="flex justify-center items-center gap-1 text-xs text-ink-muted">
                  <Clock className="size-3.5" /> Thời lượng
                </div>
                <div className="mt-1 text-xl font-bold">
                  {(latest.segment.durationMs / 1000).toFixed(2)}
                </div>
                <div className="text-[10px] text-ink-muted">giây ({latest.segment.durationMs}ms)</div>
              </div>

              <div className="rounded-lg border border-line bg-surface p-2.5">
                <div className="flex justify-center items-center gap-1 text-xs text-ink-muted">
                  <Layers className="size-3.5" /> Tốc độ
                </div>
                <div className="mt-1 text-xl font-bold">{latest.segment.fps.toFixed(1)}</div>
                <div className="text-[10px] text-ink-muted">FPS</div>
              </div>
            </div>

            <div className="rounded-md bg-surface-muted px-3 py-2 text-xs text-ink-muted flex justify-between items-center">
              <span>Định dạng khung người:</span>
              <span className={`font-medium ${latest.partial ? 'text-amber-600' : 'text-emerald-600'}`}>
                {latest.partial ? '⚠️ Bị khuất vai' : '✅ Đạt chuẩn (đủ 2 vai)'}
              </span>
            </div>
          </div>
        ) : (
          <p className="py-2 text-sm text-ink-muted text-center">
            Chưa có cử chỉ nào. Hãy bấm <strong>Start camera</strong> và thực hiện một cử chỉ tay.
          </p>
        )}
      </div>

      {/* =========================================================================
       * [BÁO CÁO TIẾN ĐỘ - BƯỚC 6: DỰ ĐOÁN TỪ ASL TẠM THỜI ĐƯỢC COMMENT]
       * Giai đoạn này chỉ báo cáo kết quả Phân đoạn cử chỉ (Bước 3).
       * Phần dự đoán từ ngữ của mô hình được bảo lưu cho giai đoạn báo cáo tiếp theo.
       * =========================================================================
      {decision && decision.top.length > 0 ? (
        <div>
          <h3 className="mb-2 text-sm font-semibold">Top guesses</h3>
          <GuessList guesses={decision.top} labels={labels} muted={decision.kind !== 'sign'} />
        </div>
      ) : null}
      ========================================================================= */}

      <div>
        <h3 className="mb-2 text-sm font-semibold flex items-center justify-between">
          <span>Lịch sử các phân đoạn cử chỉ vừa nhận</span>
          <span className="text-xs font-normal text-ink-muted">{history.length} phân đoạn</span>
        </h3>
        {history.length === 0 ? (
          <p className="text-sm text-ink-muted italic">Chưa có phân đoạn nào trong phiên này.</p>
        ) : (
          <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
            {history.map((entry, idx) => (
              <div
                key={entry.id}
                className="flex items-center justify-between rounded-lg border border-line bg-surface px-3 py-2 text-xs"
              >
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-accent">#{history.length - idx}</span>
                  <span className="text-ink-muted">{entry.timeStr}</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="font-medium">{entry.segment.frames.length} frames</span>
                  <span className="text-ink-muted">{(entry.segment.durationMs / 1000).toFixed(2)}s</span>
                  <span className="text-ink-muted">({entry.segment.fps.toFixed(0)} FPS)</span>
                  <span className={`px-1.5 py-0.5 rounded text-[10px] ${entry.partial ? 'bg-amber-100 text-amber-800' : 'bg-emerald-100 text-emerald-800'}`}>
                    {entry.partial ? 'Khuất' : 'Chuẩn'}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="rounded-lg border border-dashed border-line bg-surface/50 p-3 text-xs text-ink-muted">
        💡 <strong>Ghi chú tiến độ:</strong> Chuỗi khung xương của cử chỉ đã được phân đoạn thành công (Bước 3). Các module tiếp theo: <em>Trích xuất đặc trưng (Bước 4)</em> và <em>Mô hình nhận diện Conv-Transformer (Bước 5)</em> đang được tạm đóng mã nguồn để phục vụ báo cáo giai đoạn này.
      </div>
    </div>
  );
}
