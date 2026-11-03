/**
 * The live pipeline, independent of where frames come from (camera, Replay or
 * the e2e injection hook): frame status -> segmenter -> features -> model ->
 * decision. Inference is injected so the logic is unit-testable without
 * workers or WASM.
 */
// [BÁO CÁO TIẾN ĐỘ] Tạm thời comment các module của Bước 4, 5, 6
// import { decide, type Decision } from './decision';
import type { Decision } from './decision';
// import { modelInput, type ModelInput } from './features';
import type { ModelInput } from './features';
import type { Frame } from './landmarks';
import type { ModelMeta, SignLabel } from './meta';
import { Segmenter, frameStatus, type FrameStatus, type Segment } from './segmenter';

export interface Inference {
  logits: Float32Array;
  embedding: Float32Array;
}

export type Infer = (input: ModelInput) => Promise<Inference>;

export interface RecognitionResult {
  decision: Decision;
  inference: Inference;
  segment: Segment;
  /** True when the signer's shoulders were out of view for most of the sign. */
  partial: boolean;
}

export type RecognizerEvent =
  | { type: 'status'; status: FrameStatus }
  | { type: 'signing' }
  | { type: 'discarded' }
  | { type: 'result'; result: RecognitionResult }
  | { type: 'error'; message: string };

export interface TrackedFrame {
  frame: Frame;
  /** Milliseconds, monotonic. */
  t: number;
  /** People the pose landmarker found (0, 1 or more). */
  poses: number;
}

export class Recognizer {
  private readonly segmenter = new Segmenter();
  private statuses: FrameStatus[] = [];
  private lastStatus: FrameStatus | null = null;
  private generation = 0;

  constructor(
    _meta: ModelMeta,
    private readonly labels: readonly SignLabel[],
    _infer: Infer,
    private readonly emit: (event: RecognizerEvent) => void,
  ) {
    void _meta;
    void _infer;
  }

  get shipped(): boolean[] {
    return this.labels.map((l) => l.shipped);
  }

  reset(): void {
    this.generation++;
    this.segmenter.reset();
    this.statuses = [];
    this.lastStatus = null;
  }

  push({ frame, t, poses }: TrackedFrame): void {
    const status = frameStatus(frame, poses);
    if (status !== this.lastStatus) {
      this.lastStatus = status;
      this.emit({ type: 'status', status });
    }
    const event = this.segmenter.push(frame, t, { crowded: status === 'crowded' });
    if (this.segmenter.state === 'signing') this.statuses.push(status);
    this.handle(event);
  }

  /** End an open sign now, e.g. when a replay clip finishes. */
  flush(): void {
    this.handle(this.segmenter.flush());
  }

  private handle(event: ReturnType<Segmenter['push']>): void {
    if (!event) return;
    if (event.type === 'start') {
      this.statuses = [];
      this.emit({ type: 'signing' });
      return;
    }
    const statuses = this.statuses;
    this.statuses = [];
    if (event.type === 'discard') {
      this.emit({ type: 'discarded' });
      return;
    }
    // Vote only over frames where the signer's hands were seen: frames without
    // hands (the gap that ends a sign, short dropouts) say nothing about framing.
    const partialFrames = statuses.filter((s) => s === 'partial').length;
    const framedFrames = partialFrames + statuses.filter((s) => s === 'ok').length;
    const partial = partialFrames > framedFrames / 2;
    void this.classify(event.segment, partial, this.generation);
  }

  private async classify(segment: Segment, partial: boolean, generation: number): Promise<void> {
    /* =========================================================================
     * [BÁO CÁO TIẾN ĐỘ - DOWNGRADE XUỐNG BƯỚC 3: PHÂN ĐOẠN CỬ CHỈ TAY]
     * Các bước sau tạm thời được comment để phục vụ báo cáo tiến độ phân đoạn:
     * - BƯỚC 4: Trích xuất đặc trưng vector 460 chiều (modelInput trong features.ts)
     * - BƯỚC 5: Suy luận mô hình học sâu Conv-Transformer qua ONNX Runtime (this.infer)
     * - BƯỚC 6: Tính toán xác suất Softmax và lọc nhãn Top-3 (decide trong decision.ts)
     * =========================================================================
    const input = modelInput(segment.frames, segment.fps, this.meta);
    if (input.frames < this.meta.minFrames) {
      this.emit({ type: 'discarded' });
      return;
    }
    try {
      const inference = await this.infer(input);
      if (generation !== this.generation) return; // the mode or source changed meanwhile
      const decision = decide(inference.logits, this.shipped, this.meta.calibration, { partial });
      this.emit({ type: 'result', result: { decision, inference, segment, partial } });
    } catch (error) {
      this.emit({ type: 'error', message: error instanceof Error ? error.message : String(error) });
    }
    ========================================================================= */

    // [BƯỚC 3] Phát tín hiệu kết quả phân đoạn cử chỉ (Segment Complete)
    if (generation !== this.generation) return;
    this.emit({
      type: 'result',
      result: {
        segment,
        partial,
        // Dữ liệu mô hình ở các bước sau được để trống (mock/stub cho bước 3)
        decision: { kind: 'sign', top: [] },
        inference: { logits: new Float32Array(), embedding: new Float32Array() },
      },
    });
  }
}
