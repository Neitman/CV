import { Boxes, BrainCircuit, Gauge, ScanLine, ShieldCheck } from 'lucide-react';
import type { ModelState } from '../app/session';
import { DATASET_CREDIT, DATASET_URL, REPO_URL } from '../links';
import { Card } from './ui';

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

export function HowItWorks({ model }: { model: ModelState }) {
  const meta = model.status === 'ready' ? model.model.meta : null;
  const lips = meta?.featureSet !== 'hands_pose';
  const steps = [
    {
      icon: ScanLine,
      title: 'Bước 1 & 2: Bắt khung xương',
      badge: 'Đã hoàn thành',
      badgeColor: 'bg-emerald-100 text-emerald-800 border-emerald-200',
      body: lips
        ? 'MediaPipe Holistic trích xuất 543 điểm mốc (bàn tay, thân trên và môi) theo thời gian thực trong Web Worker.'
        : 'MediaPipe nhận diện 2 bàn tay và tư thế thân trên trong Web Worker.',
    },
    {
      icon: Boxes,
      title: 'Bước 3: Phân đoạn cử chỉ',
      badge: 'Trọng tâm báo cáo',
      badgeColor: 'bg-accent-soft text-accent border-accent/30',
      body: 'Thuật toán Segmenter theo dõi vận tốc hai bàn tay, tự động bắt đầu ghi chuỗi và chốt phân đoạn khi tay dừng yên ≥ 300ms hoặc hạ xuống.',
    },
    {
      icon: BrainCircuit,
      title: 'Bước 4: Chuẩn hoá đặc trưng',
      badge: 'Giai đoạn kế tiếp',
      badgeColor: 'bg-surface-muted text-ink-muted border-line',
      body: 'Chuẩn hoá toạ độ tâm vai, khử méo webcam ngang-dọc và đồng nhất tay thuận thành vector đặc trưng 460 chiều.',
    },
    {
      icon: Gauge,
      title: 'Bước 5 & 6: Nhận diện mô hình',
      badge: 'Giai đoạn kế tiếp',
      badgeColor: 'bg-surface-muted text-ink-muted border-line',
      body: 'Mô hình Conv-Transformer (ONNX) dự đoán 250 từ ASL với cơ chế lọc tin cậy (ngưỡng 0.29).',
    },
  ];

  return (
    <section id="how-it-works" aria-labelledby="how-heading" className="mt-16 scroll-mt-6">
      <h2 id="how-heading" className="text-2xl font-semibold tracking-tight">
        Lộ trình Pipeline & Tiến độ phát triển
      </h2>
      <p className="mt-2 max-w-2xl text-ink-muted">
        Hệ thống được thiết kế theo kiến trúc chia tầng. Hiện tại dự án đã hoàn thiện trọn vẹn đến <strong>Bước 3 (Phân đoạn cử chỉ tay)</strong>.
      </p>
      <ol className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {steps.map((step, i) => (
          <li key={step.title}>
            <Card className="h-full p-5 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between">
                  <span className="grid size-9 place-items-center rounded-lg bg-accent-soft text-accent">
                    <step.icon aria-hidden className="size-5" />
                  </span>
                  <span className={`text-[11px] font-medium px-2 py-0.5 rounded-full border ${step.badgeColor}`}>
                    {step.badge}
                  </span>
                </div>
                <div className="text-xs font-medium text-ink-muted mt-2">Giai đoạn {i + 1}</div>
                <h3 className="mt-1 font-semibold">{step.title}</h3>
                <p className="mt-1 text-sm leading-relaxed text-ink-muted">{step.body}</p>
              </div>
            </Card>
          </li>
        ))}
      </ol>

      <div className="mt-6 grid gap-4 lg:grid-cols-2">
        <Card className="p-5">
          <h3 className="font-semibold">Held-out accuracy</h3>
          {meta?.metrics ? (
            <p className="mt-1 text-sm text-ink-muted">
              On {meta.metrics.testClips.toLocaleString('en')} clips from signers the model never
              saw in training.
            </p>
          ) : null}
          {meta?.metrics ? (
            <dl className="mt-3 grid grid-cols-2 gap-3 text-sm">
              {[
                ['Top-1, all signs', pct(meta.metrics.top1)],
                ['Top-5, all signs', pct(meta.metrics.top5)],
                // When every sign ships, a shipped-set top-1 would repeat the first tile;
                // show the top-3 instead, which matches the three guesses on this page.
                meta.metrics.shippedSigns === meta.classes
                  ? ['Top-3, all signs', pct(meta.metrics.shippedTop3)]
                  : [
                      `Top-1, ${meta.metrics.shippedSigns} shipped signs`,
                      pct(meta.metrics.shippedTop1),
                    ],
                ['Teach your own (8-way, 3 takes)', pct(meta.metrics.teachAccuracy)],
              ].map(([k, v]) => (
                <div key={k} className="rounded-lg bg-surface-muted p-3">
                  <dt className="text-ink-muted">{k}</dt>
                  <dd className="mt-0.5 text-xl font-semibold tabular-nums">{v}</dd>
                </div>
              ))}
            </dl>
          ) : (
            <p className="mt-2 text-sm leading-relaxed text-ink-muted">
              Not trained yet. Numbers appear here once the model is trained and evaluated on
              signers it never saw; every figure comes from the repository’s evaluation script.
            </p>
          )}
          <p className="mt-3 text-xs text-ink-muted">
            Training data:{' '}
            <a href={DATASET_URL} className="underline underline-offset-2">
              {DATASET_CREDIT}
            </a>
            .
          </p>
        </Card>
        <Card className="p-5">
          <h3 className="flex items-center gap-2 font-semibold">
            <ShieldCheck aria-hidden className="size-5 text-success" />
            Private by design
          </h3>
          <ul className="mt-2 space-y-1.5 text-sm leading-relaxed text-ink-muted">
            <li>The camera starts only when you click, and stops when you leave the tab.</li>
            <li>Video never leaves the page. There is no server, account or analytics.</li>
            <li>
              Taught gestures are stored as numbers in this browser only, and the Clear button
              removes them.
            </li>
            <li>
              All code, models and runtimes are served from this site.{' '}
              <a href={REPO_URL} className="underline underline-offset-2">
                Source on GitHub
              </a>
              .
            </li>
          </ul>
        </Card>
      </div>
    </section>
  );
}
