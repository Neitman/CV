import { Cpu, Lock, Sparkles } from 'lucide-react';
import { useSyncExternalStore } from 'react';
import type { Session } from './app/session';
import { HowItWorks } from './components/HowItWorks';
// [BÁO CÁO TIẾN ĐỘ] Tạm thời đóng các component của Bước 5 & 6
// import { ModeTabs, type Mode } from './components/ModeTabs';
// import { PracticePanel } from './components/PracticePanel';
// import { TeachPanel } from './components/TeachPanel';
import { ModelGate } from './components/ModelGate';
import { PreviewNotice } from './components/PreviewNotice';
import { RecognisePanel } from './components/RecognisePanel';
import { SiteHeader } from './components/SiteHeader';
import { Stage } from './components/Stage';
import { Card } from './components/ui';
import { AUTHOR_URL, DATASET_CREDIT, PORTFOLIO_URL, REPO_URL } from './links';

export function App({ session }: { session: Session }) {
  const snap = useSyncExternalStore(session.subscribe, session.getSnapshot);
  const mode = 'recognise';
  const model = snap.model;
  const untrained = model.status === 'ready' && model.model.meta.metrics === null;
  // const shipped =
  //   model.status === 'ready' ? model.model.labels.filter((l) => l.shipped).length : null;

  // const changeMode = (next: Mode) => {
  //   setMode(next);
  //   session.resetRecognition();
  // };

  return (
    <>
      <a href="#demo" className="skip-link">
        Skip to the demo
      </a>
      <SiteHeader />
      {untrained ? <PreviewNotice /> : null}
      <main id="main" className="mx-auto max-w-6xl px-4 pt-8 pb-16 sm:pt-12">
        <div className="max-w-3xl">
          <h1 className="text-3xl font-semibold tracking-tight text-balance sm:text-5xl">
            Nhận diện Cử chỉ - Báo cáo Tiến độ: Bước 3
          </h1>
          <p className="mt-3 max-w-2xl text-base leading-relaxed text-ink-muted sm:text-lg">
            Hệ thống hoàn thành 3 giai đoạn đầu tiên: <strong>(1) Thu nhận hình ảnh từ Camera</strong>, <strong>(2) Trích xuất khung xương (MediaPipe Holistic)</strong> và <strong>(3) Phân đoạn cử chỉ tay (Sign Segmentation)</strong> theo thời gian thực.
          </p>
          <ul className="mt-4 flex flex-wrap gap-2 text-sm" aria-label="Highlights">
            {[
              { icon: Lock, text: 'Xử lý hoàn toàn On-device' },
              { icon: Cpu, text: 'MediaPipe Holistic WASM' },
              { icon: Sparkles, text: 'Báo cáo giai đoạn: Bước 3' },
            ].map(({ icon: Icon, text }) => (
              <li
                key={text}
                className="inline-flex items-center gap-1.5 rounded-full border border-line bg-surface px-3 py-1"
              >
                <Icon aria-hidden className="size-3.5 text-accent" />
                {text}
              </li>
            ))}
          </ul>
        </div>

        <div
          id="demo"
          className="mt-8 grid scroll-mt-4 gap-6 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]"
        >
          <Stage session={session} snap={snap} />
          <Card className="min-w-0 self-start p-4 sm:p-5">
            {/* =========================================================================
             * [BÁO CÁO TIẾN ĐỘ - BƯỚC 5 & 6: CÁC CHẾ ĐỘ NÂNG CAO TẠM ĐÓNG]
             * ModeTabs, PracticePanel và TeachPanel được tạm thời comment để chỉ báo cáo
             * tiến độ Bước 3: Phân đoạn cử chỉ tay (Sign Segmentation).
             * =========================================================================
            <ModeTabs mode={mode} onChange={changeMode} />
            ========================================================================= */}
            <div
              role="tabpanel"
              id={`panel-${mode}`}
              aria-labelledby={`tab-${mode}`}
              tabIndex={0}
              className="focus-visible:outline-offset-4"
            >
              <ModelGate
                state={model}
                onRetry={() => {
                  void session.loadModel();
                }}
              >
                {(loaded) => (
                  <RecognisePanel session={session} snap={snap} model={loaded} />
                  /*
                  mode === 'recognise' ? (
                    <RecognisePanel session={session} snap={snap} model={loaded} />
                  ) : mode === 'practice' ? (
                    <PracticePanel session={session} model={loaded} />
                  ) : (
                    <TeachPanel session={session} model={loaded} />
                  )
                  */
                )}
              </ModelGate>
              {snap.inferenceError ? (
                <p role="alert" className="mt-4 text-sm text-danger">
                  Lỗi phân tích: {snap.inferenceError}
                </p>
              ) : null}
            </div>
          </Card>
        </div>

        <HowItWorks model={model} />
      </main>
      <footer className="border-t border-line">
        <div className="mx-auto flex max-w-6xl flex-col gap-1 px-4 py-6 text-sm text-ink-muted sm:flex-row sm:justify-between">
          <p>
            GestureSign by{' '}
            <a href={AUTHOR_URL} className="underline underline-offset-2">
              Shashank Yadav
            </a>{' '}
            ·{' '}
            <a href={PORTFOLIO_URL} className="underline underline-offset-2">
              shashankyadav.dev
            </a>{' '}
            ·{' '}
            <a href={REPO_URL} className="underline underline-offset-2">
              source
            </a>{' '}
            · MIT licence
          </p>
          <p>Data: {DATASET_CREDIT}</p>
        </div>
      </footer>
    </>
  );
}
