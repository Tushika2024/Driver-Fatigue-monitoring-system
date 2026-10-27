import { api } from "../api";
import { useAsync } from "../hooks/useAsync";
import { BarList } from "../components/Charts";
import { ErrorBox, Loading, Panel, SyntheticBanner, Tag } from "../components/Ui";
import { num, pct } from "../format";

const short = (s) => s.replace(/^[A-G]\.\s*/, "").replace("OURS: ", "★ ");
const NEEDS_LABELS = (m) => /^F\./.test(m) || /^G\./.test(m);

export default function Evaluation() {
  const ev = useAsync(api.evaluation);
  if (ev.loading) return <Loading what="Loading results" />;
  if (ev.error) return <ErrorBox error={ev.error} onRetry={ev.reload} />;
  const t = ev.data.tables;
  if (!ev.data.available) return (
    <Panel title="Offline evaluation"><p className="empty">No results found in <code>{ev.data.dir}</code>. Run <code>fatigue evaluate</code> (see the backend README), then point <code>FATIGUE_RESULTS_DIR</code> at the output folder.</p></Panel>
  );
  const main = t.main_comparison?.rows || [];
  const leak = t.leakage_demo?.rows || [];
  const abl = t.ablations?.rows || [];
  const early = t.earliness?.rows || [];
  const fa = t.false_alarms_per_hour?.rows || [];
  const fair = (t.fairness?.rows || []).filter((r) => String(r.row).startsWith("GAP"));
  const people = main[0]?.people;

  return (
    <div className="evaluation">
      {ev.data.synthetic && <SyntheticBanner>These results come from {people ?? "a few"} simulated drivers. They show the evaluation harness runs; they are <b>not</b> evidence about real drivers. Re-run on UTA-RLDD for real numbers.</SyntheticBanner>}
      <Panel title="Protocol" aside={<Tag>leave-one-person-out</Tag>}>
        <p className="lede">Every number below is for a driver the model has <b>never seen</b>. Bars show each method's mean accuracy across people, and the whisker is the spread between people: with so few people the spread matters as much as the mean.</p>
      </Panel>

      <div className="two-col">
        <Panel title="Which method works best?">
          <BarList rows={main.map((r) => ({ label: short(r.method), value: r["per-person acc mean"], spread: r["per-person acc std"], tone: r.method.includes("OURS") ? "ours" : NEEDS_LABELS(r.method) ? "ghost" : "" }))} />
          <p className="muted">Hatched bars (F, G) calibrate on a labelled recording of the same person, so they are upper bounds and not deployable. The ★ row is the deployable adaptive method.</p>
        </Panel>

        <Panel title="Why the split matters">
          <BarList rows={leak.map((r) => ({ label: r.method.includes("WRONG") ? "Random split (WRONG)" : "Leave-one-person-out", value: r["acc (pooled)"], tone: r.method.includes("WRONG") ? "bad" : "ours" }))} />
          <p className="muted">A random window split puts the same person in training and test, so the model can recognise <i>who</i> it is looking at. {ev.data.synthetic && "Synthetic data cannot reproduce the size of the real-world inflation, so the gap here is small. Do not quote it."}</p>
        </Panel>

        <Panel title="Which signals matter? (ablations)">
          <BarList rows={abl.map((r) => ({ label: r.method.replace(/_/g, " "), value: r["per-person acc mean"], spread: r["per-person acc std"] }))} />
        </Panel>

        <Panel title="How early does it warn?">
          <BarList max={Math.max(30, ...early.map((r) => r.median_lead_s || 0))} fmt={(v) => `${num(v, 0)} s`}
                   rows={early.map((r) => ({ label: short(r.method), value: r.median_lead_s || 0, tone: r.method.includes("OURS") ? "ours" : "" }))} />
          <table className="tbl compact"><thead><tr><th>Method</th><th>Clips alerted before first microsleep</th></tr></thead>
            <tbody>{early.map((r, i) => <tr key={i}><td>{short(r.method)}</td><td className="mono">{pct(r.frac_alerted_before)} of {r.n_clips}</td></tr>)}</tbody></table>
          <p className="muted">Microsleep is taken as an eye-closure of at least 1 s: an assumption, set in the config.</p>
        </Panel>

        <Panel title="False alarms per hour" aside={<Tag>lower is better</Tag>}>
          <BarList max={Math.max(...fa.flatMap((r) => [r["with persistence"], r["without persistence"]]), 1)} fmt={(v) => num(v, 0)}
                   rows={fa.flatMap((r) => [
                     { label: `${short(r.method)} · with persistence`, value: r["with persistence"], tone: r.method.includes("OURS") ? "ours" : "" },
                     { label: `${short(r.method)} · without`, value: r["without persistence"], tone: "ghost" }])} />
          <p className="muted">Persistence means a signal must hold for several windows before an alert fires.</p>
        </Panel>

        <Panel title="Fairness slice" aside={<Tag tone="warn">methodology demo</Tag>}>
          {fair.length === 0 ? <p className="empty">No fairness table.</p> : (
            <table className="tbl compact">
              <thead><tr><th>Method</th><th>Attribute</th><th>Equalised-odds gap</th><th>95% CI</th></tr></thead>
              <tbody>{fair.map((r, i) => (
                <tr key={i}><td>{short(r.method)}</td><td>{r.attr.replace("_", " ")}</td>
                  <td className="mono">{num(r.eq_odds_gap, 2)}</td>
                  <td className="mono">{num(r.eq_odds_ci_lo, 2)} to {num(r.eq_odds_ci_hi, 2)}</td></tr>))}</tbody>
            </table>
          )}
          <p className="muted">Only objective attributes visible in the video (glasses, facial hair). No gender or ethnicity inference. With this few people the intervals are wide: this shows how the audit is done, not that any group is treated fairly.</p>
        </Panel>
      </div>
    </div>
  );
}
