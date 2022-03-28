let vizUpdateCBs: (() => void)[] = [];

export const registerVizUpdateCB = (cb: () => void) => vizUpdateCBs.push(cb);

export const deregisterVizUpdateCB = (cb: () => void) =>
  (vizUpdateCBs = vizUpdateCBs.filter(c => c !== cb));

export const updateViz = () => vizUpdateCBs.forEach(cb => cb());
