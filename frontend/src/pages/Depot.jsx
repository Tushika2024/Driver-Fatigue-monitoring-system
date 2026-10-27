import { useState } from "react";
import { api } from "../api";
import { useAsync } from "../hooks/useAsync";
import { Heatmap } from "../components/Charts";
import { ErrorBox, Loading, Panel, Seg, SyntheticBanner, Tile } from "../components/Ui";
import { num } from "../format";

export default function Depot() {
  const [source, setSource] = useState("synthetic");
  const d = useAsync(() => api.depot(source), [source]);
  const cells = d.data?.cells || [];
  const shown = cells.filter((c) => !c.suppressed);
  const hidden = cells.length - shown.length;
  const hot = [...shown].sort((a, b) => b.pct_time_alert - a.pct_time_alert)[0];

  return (
    <div className="depot">
      {d.data?.synthetic && <SyntheticBanner>The risk pattern in this picture was built into the simulator to demonstrate the page. It is <b>not</b> a finding.</SyntheticBanner>}
      <Panel title="Depot view · aggregate only" aside={
        <Seg label="Data source" value={source} onChange={setSource}
             options={[{ value: "synthetic", label: "Synthetic fleet" }, { value: "sessions", label: "Recorded sessions" }]} />}>
        <p className="lede">Share of monitored time at <b>MODERATE or higher</b>, by time of day and route. There is no per-driver table and no ranking.
          Any cell with fewer than <b>{d.data?.k ?? "k"}</b> contributing drivers is hidden, so a small group can't point at a person.</p>
        {d.loading && <Loading what="Loading depot data" />}
        {d.error && <ErrorBox error={d.error} onRetry={d.reload} />}
        {d.data && cells.length === 0 && <p className="empty">No recorded sessions yet. Run some from Live or Report, or switch to the synthetic fleet.</p>}
        {d.data && cells.length > 0 && <Heatmap slots={d.data.slots} routes={d.data.routes} cells={cells} k={d.data.k} />}
      </Panel>
      {d.data && cells.length > 0 && (
        <div className="tiles">
          <Tile label="Logs combined" value={d.data.n_logs} sub="shifts, no names kept" />
          <Tile label="Cells shown" value={shown.length} sub={`${hidden} hidden by the privacy floor`} />
          {hot && <Tile label="Highest cell" value={`${num(hot.pct_time_alert, 0)}%`} sub={`${hot.slot_label} · ${hot.route}`} tone="warn" />}
        </div>
      )}
      <Panel title="How to read this">
        <ul className="plain">
          <li>Use it to schedule rest breaks or rethink a route or time slot, not to judge a person.</li>
          <li>A bright cell means "more of the time at MODERATE+", which can come from the road, the hour or the roster.</li>
          <li>Hidden cells are a feature: with few drivers, an average is effectively an individual's record.</li>
        </ul>
      </Panel>
    </div>
  );
}
