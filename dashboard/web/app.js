/* pocketchange trading terminal: vanilla JS, no build step.
   Polls /api/state every 5 s and draws the equity curve
   and cached candles on <canvas>. */
"use strict";

const $id = (i) => document.getElementById(i);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

// --- formatters ---------------------------------------------------------------------------------
const nf = (v, dp) => Number(v).toLocaleString("en-US", {minimumFractionDigits: dp, maximumFractionDigits: dp});
const fUSD = (v, dp = 2) => (v == null ? "—" : "$" + nf(v, dp));
const fSign = (v, dp = 2) => {
  if (v == null) return "—";
  return (v < 0 ? "-" : "+") + "$" + nf(Math.abs(v), dp);
};
const fSignClass = (v) => (v > 0 ? "pos" : v < 0 ? "neg" : "dim");
const trimNum = (v, dp) => {
  const s = Number(v).toFixed(dp);
  return s.includes(".") ? s.replace(/0+$/, "").replace(/\.$/, "") : s;
};
const fAmt = (v) => (v == null ? "—" : trimNum(v, Math.abs(v) >= 100 ? 2 : Math.abs(v) >= 1 ? 4 : 8));
const fPrice = (v) => (v == null ? "—" : Math.abs(v) >= 1000 ? "$" + nf(v, 2) : "$" + trimNum(v, Math.abs(v) >= 1 ? 4 : 6));
const fPct = (v) => (v == null ? "—" : (v > 0 ? "+" : "") + (v * 100).toFixed(1) + "%");
const fQty = (v) => (v == null ? "—" : trimNum(v, 6));
const pad2 = (n) => String(n).padStart(2, "0");
const fTime = (iso) => {
  const d = new Date(iso);
  return `${pad2(d.getUTCMonth() + 1)}-${pad2(d.getUTCDate())} ${pad2(d.getUTCHours())}:${pad2(d.getUTCMinutes())}:${pad2(d.getUTCSeconds())}`;
};
const fHM = (ms) => { const d = new Date(ms); return `${pad2(d.getUTCHours())}:${pad2(d.getUTCMinutes())}`; };
const fMD = (ms) => { const d = new Date(ms); return `${pad2(d.getUTCMonth() + 1)}-${pad2(d.getUTCDate())}`; };

// --- state ---------------------------------------------------------------------------------------
let S = null;                 // last /api/state payload
let seriesData = null;        // last /api/series payload
let candleData = null;        // last /api/candles payload
let range = "7d";
let tab = "equity";
let hover = null;             // {x, y} in CSS pixels
let offline = false;

const RANGES = ["24h", "7d", "30d", "All"];

async function api(path, opts) {
  const r = await fetch(path, opts ? {headers: {"Content-Type": "application/json"}, ...opts} : undefined);
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r.json();
}

// --- polling loops --------------------------------------------------------------------------------
async function loopState() {
  try {
    const s = await api("/api/state");
    S = s;
    setOffline(false);
    s.ready ? renderState(s) : renderNoDb(s);
  } catch (e) {
    setOffline(true);
  } finally {
    setTimeout(loopState, 5000);
  }
}

async function loopTrades() {
  if (S && S.ready) {
    try { renderTrades(await api("/api/trades?limit=120")); } catch (e) { setOffline(true); }
  }
  setTimeout(loopTrades, 15000);
}

async function loopChart() {
  if (S && S.ready) {
    try {
      if (tab === "equity") {
        seriesData = await api(`/api/series?range=${range}`);
      } else {
        candleData = await api("/api/candles?limit=400");
      }
      drawChart();
    } catch (e) { setOffline(true); }
  }
  setTimeout(loopChart, tab === "equity" ? 15000 : 60000);
}

function setOffline(on) {
  if (offline !== on) {
    offline = on;
    $id("offline").classList.toggle("hidden", !on);
  }
}

// --- renderers -------------------------------------------------------------------------------------
function renderNoDb(s) {
  $id("nodb-text").textContent = s.message.replace(/`/g, "");
  $id("nodb").classList.remove("hidden");
  $id("layout").classList.add("hidden");
  $id("halt-banner").classList.add("hidden");
}

function renderState(s) {
  $id("nodb").classList.add("hidden");
  $id("layout").classList.remove("hidden");

  const badge = $id("mode-badge");
  badge.textContent = s.dry_run ? "DRY RUN" : "LIVE";
  badge.className = "badge " + (s.dry_run ? "dry" : "live");

  const beat = $id("hb");
  const dot = beat.querySelector(".dot");
  const age = s.heartbeat_age_s;
  let cls = "dead", text = "no signal", title = "the bot has never sent a heartbeat";
  if (age != null) {
    if (age < s.loop_interval_s * 5) { cls = "ok"; text = "bot alive"; title = `heartbeat ${Math.round(age)}s ago`; }
    else if (age < s.loop_interval_s * 15) { cls = "warn"; text = "bot slow"; title = `heartbeat ${Math.round(age / 60)}min ago`; }
    else { cls = "dead"; text = "bot stale"; title = `heartbeat ${Math.round(age / 60)}min ago — is the bot running?`; }
  }
  dot.className = "dot " + cls;
  beat.title = title;
  $id("hb-text").textContent = text;

  const banner = $id("halt-banner");
  if (s.halted && s.halt_banner) { banner.textContent = s.halt_banner; banner.classList.remove("hidden"); }
  else banner.classList.add("hidden");

  renderTicker(s);
  renderKpis(s);
  renderControls(s);
  renderPositions(s);
  renderAi(s);
  renderSniper(s);
}

function renderTicker(s) {
  const el = $id("ticker");
  el.innerHTML = (s.ticker || []).map((r) => {
    const cls = r.change == null ? "flat" : r.change >= 0 ? "up" : "down";
    return `<span class="tick" title="${esc(r.asset)}"><b>${esc(r.symbol)}</b>` +
           `<span>${fPrice(r.price)}</span><span class="chg ${cls}">${fPct(r.change)}</span></span>`;
  }).join("");
}

function kpiCell(label, value, sub, extra) {
  return `<div class="kpi"><div class="label">${label}</div><div class="value">${value}</div>` +
         (sub ? `<div class="sub">${sub}</div>` : "") + (extra || "") + `</div>`;
}

function renderKpis(s) {
  const el = $id("kpi-strip");
  const eq = s.equity, floor = s.floor;
  const fill = eq == null ? 0 : Math.max(0, Math.min(1, (eq - floor) / floor));
  const bar = eq == null ? "" :
    `<div class="bar" title="halt floor ${fUSD(floor)}"><i style="width:${(fill * 100).toFixed(0)}%"></i></div>`;
  const eqHelp = ` title="trading halts below ${fUSD(floor)}"`;
  let cells;
  if (s.sniper_active) {
    const sn = s.sniper.totals;
    cells = [
      kpiCell("Total equity (wallet + sniper)", fUSD(s.total_equity),
              `wallet ${fUSD(eq)} + sniper ${fUSD(sn.free + sn.open_value)}`, bar + eqHelp),
      kpiCell("USDC (wallet + sniper)", fUSD(s.usdc_total)),
      kpiCell("In positions", fUSD(s.invested_total), `sniper ${fUSD(sn.open_value)}`),
      kpiCell("SOL (fees, not traded)", fQty(s.sol_fees)),
      kpiCell("Mode", esc(s.mode_label)),
    ];
  } else if (s.multi) {
    cells = [
      kpiCell("Equity (USDC + positions)", fUSD(eq), null, bar + eqHelp),
      kpiCell("USDC", fUSD(s.usdc)),
      kpiCell("In positions", fUSD(s.invested)),
      kpiCell("SOL (fees, not traded)", fQty(s.sol_fees)),
      kpiCell("Mode", esc(s.mode_label)),
    ];
  } else {
    cells = [
      kpiCell("Equity (USDC + wSOL)", fUSD(eq), null, bar + eqHelp),
      kpiCell("USDC", fUSD(s.usdc)),
      kpiCell("wSOL", fQty(s.wsol), s.sol_price != null ? fUSD(s.wsol * s.sol_price) : null),
      kpiCell("SOL (fees, not traded)", fQty(s.sol_fees)),
      kpiCell("SOL price", fPrice(s.sol_price)),
    ];
  }
  el.innerHTML = cells.join("");
}

function renderControls(s) {
  const pub = s.public;
  const stop = $id("ctl-stop");
  stop.textContent = s.paused ? "RESUME" : "STOP";
  stop.className = "btn " + (s.paused ? "resume" : "stop");
  stop.disabled = pub || s.halted;
  stop.title = s.halted ? "Halted: only bot.reset_halt can clear this."
              : "Emergency stop: pauses trading now. The bot checks this right before any swap.";
  stop.onclick = async () => {
    if (!S || S.public || S.halted) return;
    try {
      await api("/api/pause", {method: "POST", body: JSON.stringify({paused: !S.paused})});
      refreshStateOnce();
    } catch (e) { setOffline(true); }
  };

  const risky = $id("ctl-risky");
  risky.textContent = s.risky ? "RISKY ON" : "RISKY";
  risky.className = "btn risky" + (s.risky ? " on" : "");
  risky.disabled = pub;
  risky.title = s.risky
    ? "Memecoins are in play. Click to go back to the top coins only."
    : "Turn on memecoin trading (BONK, WIF, TRUMP, PENGU and the simulated new-token sniper). Off: top coins only.";
  risky.onclick = async () => {
    if (!S || S.public) return;
    try {
      await api("/api/risky", {method: "POST", body: JSON.stringify({risky: !S.risky})});
      refreshStateOnce();
    } catch (e) { setOffline(true); }
  };

  const tf = $id("ctl-tf");
  tf.innerHTML = (s.tf_options || []).map((t) =>
    `<button data-tf="${esc(t)}" class="${t === s.timeframe ? "active" : ""}" ${pub ? "disabled" : ""}>${esc(t)}</button>`
  ).join("");
  tf.querySelectorAll("button").forEach((b) => {
    b.onclick = async () => {
      if (!S || S.public) return;
      try {
        await api("/api/timeframe", {method: "POST", body: JSON.stringify({timeframe: b.dataset.tf})});
        refreshStateOnce();
      } catch (e) { setOffline(true); }
    };
  });
  $id("ctl-lock").classList.toggle("hidden", !pub);
  $id("pos-mode").textContent = s.multi ? s.mode_label : "";
}

async function refreshStateOnce() {
  try {
    S = await api("/api/state");
    renderState(S);
    renderTrades(await api("/api/trades?limit=120"));
  } catch (e) { setOffline(true); }
}

function renderPositions(s) {
  const table = $id("positions"), empty = $id("pos-empty");
  const rows = s.positions || [];
  if (!rows.length) {
    table.querySelector("thead").innerHTML = "";
    table.querySelector("tbody").innerHTML = "";
    empty.classList.remove("hidden");
    return;
  }
  empty.classList.add("hidden");
  const ai = s.ai;
  const head = ["Asset", ...(ai ? ["AI p(win)"] : []), "Qty", "Price", "Value", "Avg entry", "Unrealised", "Realised"];
  table.querySelector("thead").innerHTML =
    "<tr>" + head.map((h, i) => `<th${i === 0 ? ' style="text-align:left"' : ""}>${h}</th>`).join("") + "</tr>";
  table.querySelector("tbody").innerHTML = rows.map((r) => {
    const aiCell = ai
      ? `<td>${r.p_win != null ? `<span class="pbar"><i style="width:${(r.p_win * 100).toFixed(0)}%"></i></span>${r.p_win.toFixed(2)}` : "—"}</td>`
      : "";
    return `<tr><td>${esc(r.asset)}</td>${aiCell}` +
      `<td>${fQty(r.qty)}</td><td>${fPrice(r.price)}</td><td>${fUSD(r.value)}</td>` +
      `<td>${fPrice(r.avg_entry)}</td>` +
      `<td class="${fSignClass(r.unrealised)}">${fSign(r.unrealised)}</td>` +
      `<td class="${fSignClass(r.realised)}">${fSign(r.realised)}</td></tr>`;
  }).join("");
}

function renderAi(s) {
  const panel = $id("ai-panel");
  if (!s.ai) { panel.classList.add("hidden"); return; }
  panel.classList.remove("hidden");
  $id("ai-body").innerHTML = `<div class="info-line">${esc(s.ml_caption || "")}</div>` +
    `<div class="info-line">Scores update every cycle while the model is loaded; a coin with no score yet has no clean history.</div>`;
}

function renderSniper(s) {
  const panel = $id("sniper-panel");
  if (!s.sniper_shown || !s.sniper) { panel.classList.add("hidden"); return; }
  const sn = s.sniper;
  panel.classList.remove("hidden");
  const t = sn.totals;
  const winRate = t.closed ? `${Math.round((t.wins / t.closed) * 100)}%` : "n/a";
  const totals = `<div class="sniper-totals">${t.open} open, ${t.closed} closed, win rate ${winRate}, ` +
    `realised <span class="${fSignClass(t.realised)}">${fSign(t.realised)}</span>, ` +
    `unrealised <span class="${fSignClass(t.unrealised)}">${fSign(t.unrealised)}</span>, ` +
    `bankroll free ${fUSD(t.free)}</div>`;
  const openTable = sn.open.length ? tableHtml(
    ["token", "age (min)", "entry $", "value $", "P&L %"],
    sn.open.map((r) => [
      esc(r.symbol), r.age_min.toFixed(1), fUSD(r.entry), r.value == null ? "—" : fUSD(r.value),
      `<span class="${fSignClass(r.pnl_pct)}">${fPct(r.pnl_pct / 100)}</span>`,
    ])) : "";
  const closedTable = sn.closed.length ? tableHtml(
    ["token", "closed", "entry $", "exit $", "P&L %", "reason"],
    sn.closed.slice(0, 12).map((r) => [
      esc(r.symbol), fTime(r.closed_at), fUSD(r.entry), fUSD(r.exit),
      `<span class="${fSignClass(r.pnl_pct)}">${fPct(r.pnl_pct / 100)}</span>`, esc(r.reason),
    ])) : "";
  $id("sniper-body").innerHTML = `<div class="info-line">${esc(sn.caption)}</div>` + totals +
    (sn.active ? openTable + closedTable : "");
}

function tableHtml(head, rows) {
  return `<table><thead><tr>${head.map((h, i) =>
    `<th${i === 0 ? ' style="text-align:left"' : ""}>${h}</th>`).join("")}</tr></thead><tbody>` +
    rows.map((r) => `<tr>${r.map((c, i) => `<td${i === 0 ? ' style="text-align:left"' : ""}>${c}</td>`).join("")}</tr>`).join("") +
    `</tbody></table>`;
}

function renderTrades(data) {
  const table = $id("trade-log"), empty = $id("log-empty");
  const rows = data.rows || [];
  $id("log-note").textContent = rows.length ? `last ${rows.length}` : "";
  if (!rows.length) {
    table.querySelector("thead").innerHTML = "";
    table.querySelector("tbody").innerHTML = "";
    empty.classList.remove("hidden");
    return;
  }
  empty.classList.add("hidden");
  const showAsset = (S && S.multi) || rows.some((r) => r.symbol !== "SOL");
  const head = ["Time", ...(showAsset ? ["Asset"] : []), "Side", "In", "Out", "Price", "Mode", "Tx"];
  table.querySelector("thead").innerHTML =
    "<tr>" + head.map((h, i) => `<th${i === 0 ? ' style="text-align:left"' : ""}>${h}</th>`).join("") + "</tr>";
  table.querySelector("tbody").innerHTML = rows.map((r) => {
    const asset = showAsset ? `<td>${esc(r.asset)}</td>` : "";
    const side = r.side === "buy"
      ? `<span class="side side-buy">BUY</span>`
      : `<span class="side side-sell">SELL</span>`;
    const mode = r.simulated ? `<span class="chip">SIM</span>` : `<span class="chip live">LIVE</span>`;
    const tx = r.tx ? `<a href="${esc(r.tx)}" target="_blank" rel="noopener">Solscan</a>` : "—";
    return `<tr><td>${fTime(r.timestamp)}</td>${asset}<td>${side}</td>` +
      `<td>${fAmt(r.amount_in)}</td><td>${fAmt(r.amount_out)}</td><td>${fPrice(r.price)}</td>` +
      `<td>${mode}</td><td>${tx}</td></tr>`;
  }).join("");
}

// --- chart -----------------------------------------------------------------------------------------
const CANVAS = $id("chart");
const WRAP = $id("chart-wrap");
const TOOLTIP = $id("tooltip");
const CSS = getComputedStyle(document.documentElement);
const C = {
  text: "#8a93a6", faint: "#5c6478", grid: "rgba(255,255,255,.05)",
  green: "#26a69a", red: "#ef5350", blue: "#4f83e6", amber: "#fbc02d",
};

function ctx2d() {
  const dpr = window.devicePixelRatio || 1;
  const w = WRAP.clientWidth, h = WRAP.clientHeight;
  if (CANVAS.width !== Math.round(w * dpr) || CANVAS.height !== Math.round(h * dpr)) {
    CANVAS.width = Math.round(w * dpr);
    CANVAS.height = Math.round(h * dpr);
  }
  const ctx = CANVAS.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return {ctx, w, h};
}

function frame(w, h) {
  return {l: 10, r: 78, t: 14, b: 24, w, h};
}

function axisGrid(ctx, P, lo, hi, fmtV, n = 4) {
  ctx.font = "10.5px 'Share Tech Mono', monospace";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  for (let i = 0; i <= n; i++) {
    const v = lo + ((hi - lo) * i) / n;
    const y = P.t + (P.h - P.t - P.b) * (1 - i / n);
    ctx.strokeStyle = C.grid;
    ctx.beginPath();
    ctx.moveTo(P.l, y);
    ctx.lineTo(P.w - P.r, y);
    ctx.stroke();
    ctx.fillStyle = C.faint;
    ctx.fillText(fmtV(v), P.w - P.r + 6, y);
  }
}

function timeAxis(ctx, P, ts) {
  const t0 = ts[0], t1 = ts[ts.length - 1], span = Math.max(1, t1 - t0);
  const spanH = span / 36e5;
  const fmt = spanH < 48 ? fHM : fMD;
  ctx.font = "10.5px 'Share Tech Mono', monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let i = 0; i <= 4; i++) {
    const t = t0 + span * (i / 4);
    const x = P.l + (P.w - P.l - P.r) * (i / 4);
    ctx.fillStyle = C.faint;
    ctx.fillText(fmt(t), x, P.h - P.b + 6);
  }
}

function lastPill(ctx, x, y, text, color) {
  ctx.font = "10.5px 'Share Tech Mono', monospace";
  const wpx = ctx.measureText(text).width + 10;
  const px = Math.min(x + 4, CANVAS.clientWidth - wpx - 2);
  ctx.fillStyle = color;
  roundRect(ctx, px, y - 8, wpx, 16, 3);
  ctx.fill();
  ctx.fillStyle = "#0b0e14";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillText(text, px + 5, y);
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function polyline(ctx, pts) {
  ctx.beginPath();
  pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.stroke();
}

function drawChart() {
  if (tab === "equity") drawEquity();
  else drawCandles();
}

function drawEquity() {
  const {ctx, w, h} = ctx2d();
  const P = frame(w, h);
  const empty = $id("chart-empty");
  const pts = (seriesData && seriesData.points) || [];
  if (!pts.length) {
    empty.textContent = S && S.ready ? "waiting for balance snapshots…" : "";
    empty.classList.remove("hidden");
    TOOLTIP.classList.add("hidden");
    return;
  }
  empty.classList.add("hidden");

  const ts = pts.map((p) => Date.parse(p.t));
  const t0 = ts[0], t1 = ts[ts.length - 1];
  const hasTotal = pts.some((p) => p.total != null);
  const vals = [seriesData.floor];
  pts.forEach((p) => { vals.push(p.equity); if (p.total != null) vals.push(p.total); });
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (hi === lo) { hi += 1; lo -= 1; }
  const pad = (hi - lo) * 0.08;
  lo -= pad; hi += pad;

  const X = (t) => P.l + (P.w - P.l - P.r) * ((t - t0) / Math.max(1, t1 - t0));
  const Y = (v) => P.t + (P.h - P.t - P.b) * (1 - (v - lo) / (hi - lo));
  const fmtAxis = (v) => "$" + (Math.abs(v) >= 1000 ? nf(v, 0) : nf(v, 2));

  axisGrid(ctx, P, lo, hi, fmtAxis);

  // the halt floor: the wallet only (the sniper's money never props it up)
  const fy = Y(seriesData.floor);
  ctx.save();
  ctx.strokeStyle = C.red;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(P.l, fy);
  ctx.lineTo(P.w - P.r, fy);
  ctx.stroke();
  ctx.restore();

  const walletPts = pts.map((p, i) => [X(ts[i]), Y(p.equity)]);
  if (hasTotal) {
    const totalPts = pts.map((p, i) => [X(ts[i]), Y(p.total != null ? p.total : p.equity)]);
    const grad = ctx.createLinearGradient(0, P.t, 0, P.h - P.b);
    grad.addColorStop(0, "rgba(38,166,154,.14)");
    grad.addColorStop(1, "rgba(38,166,154,0)");
    ctx.fillStyle = grad;
    ctx.beginPath();
    totalPts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.lineTo(P.w - P.r, P.h - P.b);
    ctx.lineTo(P.l, P.h - P.b);
    ctx.closePath();
    ctx.fill();
    ctx.strokeStyle = C.green;
    ctx.lineWidth = 1.8;
    polyline(ctx, totalPts);
    ctx.strokeStyle = "rgba(79,131,230,.75)";
    ctx.lineWidth = 1.2;
    ctx.setLineDash([3, 3]);
    polyline(ctx, walletPts);
    ctx.setLineDash([]);
  } else {
    const grad = ctx.createLinearGradient(0, P.t, 0, P.h - P.b);
    grad.addColorStop(0, "rgba(79,131,230,.16)");
    grad.addColorStop(1, "rgba(79,131,230,0)");
    ctx.fillStyle = grad;
    ctx.beginPath();
    walletPts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.lineTo(P.w - P.r, P.h - P.b);
    ctx.lineTo(P.l, P.h - P.b);
    ctx.closePath();
    ctx.fill();
    ctx.strokeStyle = C.blue;
    ctx.lineWidth = 1.8;
    polyline(ctx, walletPts);
  }

  lastPill(ctx, X(t1), Y(pts[pts.length - 1].equity), fmtAxis(pts[pts.length - 1].equity), hasTotal ? "#3d4b63" : C.blue);
  if (hasTotal) lastPill(ctx, X(t1), Y(pts[pts.length - 1].total), fmtAxis(pts[pts.length - 1].total), C.green);

  timeAxis(ctx, P, ts);

  // crosshair
  if (hover && hover.x >= P.l && hover.x <= P.w - P.r) {
    let idx = 0, best = Infinity;
    pts.forEach((_, i) => {
      const d = Math.abs(X(ts[i]) - hover.x);
      if (d < best) { best = d; idx = i; }
    });
    const cx = X(ts[idx]);
    ctx.strokeStyle = "rgba(255,255,255,.18)";
    ctx.beginPath();
    ctx.moveTo(cx, P.t);
    ctx.lineTo(cx, P.h - P.b);
    ctx.stroke();
    const base = pts[idx].equity;
    const chg = base / pts[0].equity - 1;
    let html = `<div class="t">${fTime(pts[idx].t)} UTC</div>` +
      `<div>wallet <span class="${fSignClass(chg)}">${fUSD(base)}</span> (${fPct(chg)})</div>`;
    if (pts[idx].total != null) {
      const tchg = pts[idx].total / pts[0].total - 1;
      html += `<div>total <span class="${fSignClass(tchg)}">${fUSD(pts[idx].total)}</span> (${fPct(tchg)})</div>`;
    }
    showTooltip(cx, Y(base), html, P);
  } else {
    TOOLTIP.classList.add("hidden");
  }

  $id("chart-caption").textContent = S && S.floor_caption ? S.floor_caption : "";
}

function drawCandles() {
  const {ctx, w, h} = ctx2d();
  const P = frame(w, h);
  const empty = $id("chart-empty");
  const cs = (candleData && candleData.candles) || [];
  if (!cs.length) {
    empty.textContent = "no candle cache yet — run: python -m bot.data";
    empty.classList.remove("hidden");
    TOOLTIP.classList.add("hidden");
    $id("chart-caption").textContent = `SOL ${candleData ? candleData.timeframe : ""} candles come from the local cache; the terminal never touches the network.`;
    return;
  }
  empty.classList.add("hidden");

  const VOL = 54;
  const n = cs.length;
  const ts = cs.map((c) => Date.parse(c.t));
  const t0 = ts[0], t1 = ts[ts.length - 1];
  const priceH = P.h - P.t - P.b - VOL - 10;
  const lo0 = Math.min(...cs.map((c) => c.l)), hi0 = Math.max(...cs.map((c) => c.h));
  const padP = (hi0 - lo0) * 0.06 || hi0 * 0.01;
  const lo = lo0 - padP, hi = hi0 + padP;
  const vmax = Math.max(...cs.map((c) => c.v)) || 1;

  const cw = (P.w - P.l - P.r) / n;
  const X = (i) => P.l + cw * (i + 0.5);
  const Y = (v) => P.t + priceH * (1 - (v - lo) / (hi - lo));
  const VY = (v) => P.h - P.b - (v / vmax) * VOL;
  const fmtAxis = (v) => "$" + (Math.abs(v) >= 1000 ? nf(v, 0) : trimNum(v, 2));

  axisGrid(ctx, P, lo, hi, fmtAxis, 5);

  cs.forEach((c, i) => {
    const up = c.c >= c.o;
    const color = up ? C.green : C.red;
    const x = X(i);
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, Y(c.h));
    ctx.lineTo(x, Y(c.l));
    ctx.stroke();
    const bodyTop = Y(Math.max(c.o, c.c)), bodyBot = Y(Math.min(c.o, c.c));
    const bw = Math.max(1, cw * 0.62);
    ctx.fillRect(x - bw / 2, bodyTop, bw, Math.max(1, bodyBot - bodyTop));
    ctx.globalAlpha = 0.45;
    ctx.fillRect(x - bw / 2, VY(c.v), bw, P.h - P.b - VY(c.v));
    ctx.globalAlpha = 1;
  });

  timeAxis(ctx, P, ts);

  $id("chart-caption").textContent =
    `SOL ${candleData.timeframe} candles from the local cache (${cs.length} candles; run python -m bot.data to refresh).`;

  if (hover && hover.x >= P.l && hover.x <= P.w - P.r) {
    const idx = Math.max(0, Math.min(n - 1, Math.floor((hover.x - P.l) / cw)));
    const c = cs[idx];
    const cx = X(idx);
    ctx.strokeStyle = "rgba(255,255,255,.18)";
    ctx.beginPath();
    ctx.moveTo(cx, P.t);
    ctx.lineTo(cx, P.h - P.b);
    ctx.stroke();
    const cls = (v) => (v >= 0 ? "pos" : "neg");
    const chg = c.c / c.o - 1;
    showTooltip(cx, Y(c.h), `<div class="t">${fTime(c.t)} UTC</div>` +
      `<div>O ${fmtAxis(c.o)}  H <span class="pos">${fmtAxis(c.h)}</span></div>` +
      `<div>L <span class="neg">${fmtAxis(c.l)}</span>  C ${fmtAxis(c.c)}</div>` +
      `<div>V ${trimNum(c.v, 2)}  <span class="${cls(chg)}">${fPct(chg)}</span></div>`, P);
  } else {
    TOOLTIP.classList.add("hidden");
  }
}

function showTooltip(x, y, html, P) {
  TOOLTIP.innerHTML = html;
  TOOLTIP.classList.remove("hidden");
  const tw = TOOLTIP.offsetWidth, th = TOOLTIP.offsetHeight;
  let left = x + 14;
  if (left + tw > P.w - P.r) left = x - tw - 14;
  let top = y + 14;
  if (top + th > P.h - P.b) top = y - th - 14;
  TOOLTIP.style.left = `${Math.max(2, left)}px`;
  TOOLTIP.style.top = `${Math.max(2, top)}px`;
}

// --- wiring ------------------------------------------------------------------------------------------
function bindStatic() {
  $id("chart-tabs").querySelectorAll(".tab").forEach((b) => {
    b.onclick = () => {
      tab = b.dataset.tab;
      $id("chart-tabs").querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === b));
      hover = null;
      drawChart();
    };
  });
  const pills = $id("range-pills");
  pills.innerHTML = RANGES.map((r) => `<button class="pill ${r === range ? "active" : ""}" data-r="${r}">${r}</button>`).join("");
  pills.querySelectorAll(".pill").forEach((b) => {
    b.onclick = () => {
      range = b.dataset.r;
      pills.querySelectorAll(".pill").forEach((x) => x.classList.toggle("active", x === b));
      if (tab === "equity") loopChartNow();
    };
  });
  CANVAS.addEventListener("mousemove", (e) => {
    const r = CANVAS.getBoundingClientRect();
    hover = {x: e.clientX - r.left, y: e.clientY - r.top};
    drawChart();
  });
  CANVAS.addEventListener("mouseleave", () => { hover = null; drawChart(); });
  new ResizeObserver(() => drawChart()).observe(WRAP);
}

async function loopChartNow() {
  try {
    seriesData = await api(`/api/series?range=${range}`);
    drawChart();
  } catch (e) { setOffline(true); }
}

function tickClock() {
  const d = new Date();
  $id("clock").textContent =
    `${pad2(d.getUTCHours())}:${pad2(d.getUTCMinutes())}:${pad2(d.getUTCSeconds())} UTC`;
}

window.addEventListener("DOMContentLoaded", () => {
  bindStatic();
  loopState();
  loopTrades();
  loopChart();
  tickClock();
  setInterval(tickClock, 1000);
});
