import { useCallback, useEffect, useState } from "react";

/** Load once (and on demand): { data, error, loading, reload }. */
export function useAsync(fn, deps = []) {
  const [st, setSt] = useState({ data: null, error: null, loading: true });
  const run = useCallback(() => {
    setSt((s) => ({ ...s, loading: true, error: null }));
    return fn().then(
      (data) => setSt({ data, error: null, loading: false }),
      (e) => setSt({ data: null, error: e.message || String(e), loading: false }),
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  useEffect(() => { run(); }, [run]);
  return { ...st, reload: run };
}
