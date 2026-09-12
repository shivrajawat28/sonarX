import { useQuery } from "@tanstack/react-query";
import {
  EvalModelProvenance,
  getModel,
  getModelMetrics,
  listModels,
} from "../api/client";
import ErrorBox from "../components/ErrorBox";

/**
 * Model info + HONEST metrics. Metrics come exclusively from stored
 * EvaluationRuns — if none exist, the UI says so instead of inventing numbers
 * (Section 12: "no fabricated metrics"; test asserts the 404 path).
 *
 * The detail/metrics panels follow the model ACTUALLY LOADED by the backend
 * (loaded_version), not the oldest registry entry (bugfix: listModels returns
 * entries sorted oldest-first, so versions[0] was the retired stub).
 */
export default function ModelsPage() {
  const modelsQ = useQuery({ queryKey: ["models"], queryFn: listModels });
  const versions = modelsQ.data?.items ?? [];
  const loaded = modelsQ.data?.loaded_version ?? null;
  const active =
    loaded ??
    versions.find((m) => m.status === "active")?.model_version ??
    versions[0]?.model_version ??
    null;

  const detailQ = useQuery({
    queryKey: ["model", active],
    queryFn: () => getModel(active as string),
    enabled: active != null,
  });
  const metricsQ = useQuery({
    queryKey: ["metrics", active],
    queryFn: () => getModelMetrics(active as string),
    enabled: active != null,
  });

  if (modelsQ.isLoading) return <p className="muted">loading models…</p>;
  if (modelsQ.error) return <ErrorBox error={modelsQ.error} />;

  return (
    <div>
      <div className="panel">
        <h2>Registered models</h2>
        {loaded && (
          <p className="muted">
            currently loaded for inference: <strong>{loaded}</strong>
          </p>
        )}
        <table>
          <thead>
            <tr>
              <th>Version</th>
              <th>Family</th>
              <th>Framework</th>
              <th>Status</th>
              <th>Classes</th>
            </tr>
          </thead>
          <tbody>
            {versions.map((m) => (
              <tr
                key={m.model_version}
                style={m.model_version === loaded ? { fontWeight: 600 } : undefined}
              >
                <td>
                  {m.model_version}
                  {m.model_version === loaded && (
                    <span className="muted"> ← loaded</span>
                  )}
                </td>
                <td className="muted">{m.architecture_family}</td>
                <td className="muted">{m.framework}</td>
                <td>{m.status}</td>
                <td className="muted">{Object.values(m.class_map).join(", ")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {detailQ.data && (
        <div className="panel" style={{ marginTop: "1rem" }}>
          <h2>{detailQ.data.model_version}</h2>
          <p className="muted">
            input {detailQ.data.input_size.join("×")} · {detailQ.data.notes ?? "no notes"}
          </p>
          <h3 style={{ fontSize: "0.95rem" }}>Class map (from model metadata)</h3>
          <table>
            <tbody>
              {Object.entries(detailQ.data.class_map).map(([id, name]) => (
                <tr key={id}>
                  <td className="muted">{id}</td>
                  <td>{name}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="panel" style={{ marginTop: "1rem" }}>
        <h2>Evaluation metrics</h2>
        {metricsQ.isLoading && <p className="muted">checking…</p>}
        {metricsQ.data === null && (
          <p className="muted">
            No evaluation runs recorded for this model yet. Metrics appear here
            only after a real evaluation against a versioned dataset split —
            nothing is estimated.
          </p>
        )}
        {metricsQ.data && (
          <>
            <p className="muted">
              eval run <code>{metricsQ.data.eval_run_id}</code> · recorded{" "}
              {metricsQ.data.timestamp?.slice(0, 19).replace("T", " ")} UTC ·
              dataset <code>{metricsQ.data.dataset_ref?.sha256?.slice(0, 12)}…</code>
            </p>
            <p>
              <strong>
                Split: {(metricsQ.data.split ?? "—").toUpperCase()}
              </strong>{" "}
              <span className="muted">
                (
                {metricsQ.data.split === "test"
                  ? "held-out TEST split — not used during training"
                  : metricsQ.data.split === "val"
                    ? "VALIDATION split — used for training-time model selection; NOT test performance"
                    : "TRAIN split — training data, never report as generalization"}
                )
              </span>
            </p>
            {metricsQ.data.filter_enabled != null && (
              <p className="muted">
                filtering during this evaluation:{" "}
                {metricsQ.data.filter_enabled ? "ON" : "OFF (detector-only)"}
              </p>
            )}
            <h3 style={{ fontSize: "0.95rem" }}>Overall</h3>
            <table>
              <thead>
                <tr>
                  <th>Metric</th>
                  <th>Value</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(metricsQ.data.metrics)
                  .filter(([, v]) => v != null)
                  .map(([k, v]) => (
                    <tr key={k}>
                      <td>{k}</td>
                      <td>{typeof v === "number" ? (v as number).toFixed(4) : String(v)}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
            {metricsQ.data.per_class &&
              Object.keys(metricsQ.data.per_class).length > 0 && (
                <>
                  <h3 style={{ fontSize: "0.95rem", marginTop: "1rem" }}>
                    Per-class
                  </h3>
                  <table>
                    <thead>
                      <tr>
                        <th>Class</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1</th>
                        <th>AP50</th>
                        <th>Support</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(metricsQ.data.per_class).map(
                        ([cls, m]) => (
                          <tr key={cls}>
                            <td>{cls}</td>
                            <td>{fmt(m.precision)}</td>
                            <td>{fmt(m.recall)}</td>
                            <td>{fmt(m.f1)}</td>
                            <td>{fmt(m.ap50)}</td>
                            <td className="muted">{m.support ?? "—"}</td>
                          </tr>
                        ),
                      )}
                    </tbody>
                  </table>
                  <p className="muted">
                    ghost_net is 100% synthetic — its numbers are
                    synthetic-on-synthetic and not a field capability.
                  </p>
                </>
              )}

            {metricsQ.data.confusion && (
              <ConfusionTable matrix={metricsQ.data.confusion} />
            )}

            {metricsQ.data.model && (
              <ModelProvenance model={metricsQ.data.model} />
            )}
          </>
        )}
        <ErrorBox error={metricsQ.error} />
      </div>
    </div>
  );
}

function fmt(v: number | undefined | null): string {
  return typeof v === "number" ? v.toFixed(4) : "—";
}

/**
 * Confusion matrix as recorded by the eval run (rows = ground-truth class,
 * columns = predicted class). Rendered only when the artifact was stored.
 */
function ConfusionTable({ matrix }: { matrix: Record<string, Record<string, number>> }) {
  const gts = Object.keys(matrix);
  const cols = Array.from(
    new Set(gts.flatMap((g) => Object.keys(matrix[g] ?? {}))),
  );
  if (gts.length === 0 || cols.length === 0) return null;
  return (
    <>
      <h3 style={{ fontSize: "0.95rem", marginTop: "1rem" }}>Confusion matrix</h3>
      <p className="muted" style={{ fontSize: "0.8rem" }}>
        recorded with this evaluation run (rows = ground truth, columns = predicted)
      </p>
      <table>
        <thead>
          <tr>
            <th>true \ pred</th>
            {cols.map((c) => (
              <th key={c}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {gts.map((g) => (
            <tr key={g}>
              <td>{g}</td>
              {cols.map((c) => {
                const v = matrix[g]?.[c] ?? 0;
                const correct = g === c;
                return (
                  <td
                    key={c}
                    className="muted"
                    style={correct ? { fontWeight: 600 } : undefined}
                  >
                    {v}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

/** Training provenance straight from the registry entry — never inferred. */
function ModelProvenance({ model }: { model: EvalModelProvenance }) {
  const trainCfg = model.train_config ?? {};
  return (
    <>
      <h3 style={{ fontSize: "0.95rem", marginTop: "1rem" }}>
        Model &amp; training provenance
      </h3>
      <table>
        <tbody>
          <tr>
            <td className="muted">Architecture</td>
            <td>
              {model.architecture_family} ({model.framework})
            </td>
          </tr>
          <tr>
            <td className="muted">Input size</td>
            <td>{model.input_size.join("×")}</td>
          </tr>
          <tr>
            <td className="muted">Weights</td>
            <td>
              <code>{model.checkpoint_path}</code>
            </td>
          </tr>
          <tr>
            <td className="muted">Training dataset</td>
            <td>
              <code>{model.train_dataset_ref?.path ?? "not recorded"}</code>
            </td>
          </tr>
          <tr>
            <td className="muted">Training config</td>
            <td className="muted">
              {Object.entries(trainCfg).length
                ? Object.entries(trainCfg)
                    .map(([k, v]) => `${k}=${String(v)}`)
                    .join(" · ")
                : "not recorded"}
            </td>
          </tr>
          <tr>
            <td className="muted">Registry status</td>
            <td>{model.status}</td>
          </tr>
        </tbody>
      </table>
    </>
  );
}
