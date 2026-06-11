/* FinAgent dashboard — talks to the FastAPI server in server/app.py */

const API = "";
const SESSION = "web-" + Math.random().toString(36).slice(2, 10);

const $ = (id) => document.getElementById(id);

const fmtMoney = (v) => {
  const sign = v < 0 ? "−" : "";
  const a = Math.abs(v);
  if (a >= 1_000_000) return `${sign}$${(a / 1_000_000).toFixed(2)}M`;
  if (a >= 10_000) return `${sign}$${(a / 1000).toFixed(1)}k`;
  return `${sign}$${a.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
};
const fmtFull = (v) =>
  `${v < 0 ? "−" : ""}$${Math.abs(v).toLocaleString(undefined, { maximumFractionDigits: 0 })}`;

/* ------------------------------------------------------------ action metadata */
const ACTION_GROUPS = [
  {
    label: "Debt",
    actions: [
      { id: "pay_credit_card", name: "Pay credit card", amount: true,
        hint: "Pay down credit card debt (3%/mo interest). Lowers utilization, boosts credit." },
      { id: "pay_personal_loan", name: "Pay loan", amount: true,
        hint: "Pay down the personal loan (1%/mo interest)." },
    ],
  },
  {
    label: "Invest",
    actions: [
      { id: "invest_stocks", name: "Stocks", amount: true, hint: "High growth in bull markets, painful in crashes." },
      { id: "invest_mutual_funds", name: "Mutual funds", amount: true, hint: "Tracks stocks at 0.8x — smoother ride." },
      { id: "invest_crypto", name: "Crypto", amount: true, hint: "Wild: −40% to +40% in a single month." },
      { id: "invest_bonds", name: "Bonds", amount: true, hint: "Steady. Actually rallies in bear markets." },
      { id: "invest_fd", name: "Fixed deposit", amount: true, hint: "Guaranteed 2%/mo. Boring, safe." },
      { id: "invest_commodities", name: "Commodities", amount: true, hint: "Inflation hedge — shines when markets fall." },
    ],
  },
  {
    label: "Sell",
    actions: [
      { id: "sell_stocks", name: "Stocks", amount: true, hint: "Sell stocks back to cash. Amount 0 = sell all." },
      { id: "sell_mutual_funds", name: "Mutual funds", amount: true, hint: "Sell mutual funds. Amount 0 = sell all." },
      { id: "sell_crypto", name: "Crypto", amount: true, hint: "Sell crypto. Amount 0 = sell all." },
      { id: "sell_bonds", name: "Bonds", amount: true, hint: "Sell bonds. Amount 0 = sell all." },
      { id: "sell_fd", name: "Fixed deposit", amount: true, hint: "Break the FD. Amount 0 = sell all." },
      { id: "sell_commodities", name: "Commodities", amount: true, hint: "Sell commodities. Amount 0 = sell all." },
    ],
  },
  {
    label: "Real Estate",
    actions: [
      { id: "buy_real_estate", name: "Buy property", amount: true,
        hint: "Min $50k. Earns 1%/mo rent minus 0.2%/mo maintenance, and appreciates with the market." },
      { id: "sell_real_estate", name: "Sell property", amount: true,
        hint: "Liquidate property at a 5% transaction fee. Amount 0 = sell all." },
    ],
  },
  {
    label: "Safety Net",
    actions: [
      { id: "build_emergency_fund", name: "Build emergency fund", amount: true,
        hint: "Move savings into the emergency fund. Target: 3x fixed expenses ($60k)." },
      { id: "withdraw_emergency_fund", name: "Withdraw", amount: true,
        hint: "Pull cash back out of the emergency fund. Amount 0 = withdraw all." },
    ],
  },
  {
    label: "Lifestyle",
    actions: [
      { id: "reduce_spending", name: "Cut spending", amount: false,
        hint: "Trim variable expenses by 10% (floor $4k/mo)." },
      { id: "hold", name: "Hold", amount: false,
        hint: "Do nothing this month and let the portfolio ride." },
    ],
  },
];
const ACTION_META = Object.fromEntries(
  ACTION_GROUPS.flatMap((g) => g.actions.map((a) => [a.id, a]))
);

const EVENT_TEXT = {
  none: null,
  medical: "🏥 Medical emergency — an unexpected bill hit your accounts.",
  car_repair: "🚗 Car repair — the mechanic sends his regards.",
  home_repair: "🔧 Home repair — something broke, and it wasn't cheap.",
  job_loss: "💼 Job loss — no salary next month. Brace yourself.",
  bonus: "🎉 Bonus! A windfall just landed in your savings.",
  tax_refund: "🧾 Tax refund — the government gave some back.",
  salary_raise: "📈 Salary raise — your income just went up permanently.",
};

/* ------------------------------------------------------------ state */
let tasks = [];
let selectedTask = null;
let obs = null;
let selectedAction = null;
let netWorthHistory = [];
let rewards = [];
let prevNetWorth = null;
let stepping = false;

let chartNet = null;
let chartPortfolio = null;

/* ------------------------------------------------------------ start screen */
async function loadTasks() {
  const res = await fetch(`${API}/tasks`);
  const data = await res.json();
  tasks = data.tasks;
  const grid = $("task-grid");
  grid.innerHTML = "";
  for (const t of tasks) {
    const card = document.createElement("button");
    card.className = "task-card";
    card.innerHTML = `
      <span class="diff-badge diff-${t.difficulty}">${t.difficulty}</span>
      <h3>${t.name}</h3>
      <p>${t.description}</p>
      <span class="task-goal">◎ ${t.goal}</span>`;
    card.addEventListener("click", () => {
      document.querySelectorAll(".task-card").forEach((c) => c.classList.remove("selected"));
      card.classList.add("selected");
      selectedTask = t;
      const btn = $("btn-start");
      btn.disabled = false;
      btn.textContent = `Start: ${t.name}`;
    });
    grid.appendChild(card);
  }
}

async function startEpisode() {
  const seedRaw = $("seed-input").value;
  const params = new URLSearchParams({ session_id: SESSION });
  if (selectedTask.id !== "free_play") params.set("task_id", selectedTask.id);
  if (seedRaw !== "") params.set("seed", seedRaw);

  const res = await fetch(`${API}/reset?${params}`, { method: "POST" });
  if (!res.ok) return toast(`Reset failed: ${(await res.json()).detail}`, true);
  obs = await res.json();

  netWorthHistory = [obs.net_worth];
  rewards = [];
  prevNetWorth = null;
  selectedAction = null;
  $("activity-log").innerHTML = "";
  $("reward-chips").innerHTML = "";
  $("event-banner").hidden = true;
  $("results-modal").hidden = true;
  $("btn-step").disabled = true;
  $("action-hint").textContent = "Pick an action to see what it does.";
  document.querySelectorAll(".action-chip").forEach((c) => c.classList.remove("selected"));

  $("hud-task-name").textContent = selectedTask.name;
  const diff = $("hud-task-diff");
  diff.textContent = selectedTask.difficulty;
  diff.className = `diff-badge diff-${selectedTask.difficulty}`;

  initCharts();
  render();
  logLine(`Episode started — <strong>${selectedTask.name}</strong>. ${selectedTask.goal}`, "event");

  $("screen-start").classList.remove("active");
  $("screen-game").classList.add("active");
}

/* ------------------------------------------------------------ action panel */
function buildActionPanel() {
  const wrap = $("action-groups");
  wrap.innerHTML = "";
  for (const group of ACTION_GROUPS) {
    const div = document.createElement("div");
    div.innerHTML = `<div class="action-group-label">${group.label}</div>`;
    const row = document.createElement("div");
    row.className = "action-chip-row";
    for (const a of group.actions) {
      const chip = document.createElement("button");
      chip.className = "action-chip";
      chip.textContent = a.name;
      chip.dataset.action = a.id;
      chip.addEventListener("click", () => selectAction(a, chip));
      row.appendChild(chip);
    }
    div.appendChild(row);
    wrap.appendChild(div);
  }
}

function selectAction(a, chip) {
  selectedAction = a;
  document.querySelectorAll(".action-chip").forEach((c) => c.classList.remove("selected"));
  chip.classList.add("selected");
  $("action-hint").textContent = a.hint;
  $("amount-row").style.opacity = a.amount ? "1" : "0.35";
  $("amount-input").disabled = !a.amount;
  $("btn-step").disabled = false;
  if (a.id === "buy_real_estate") $("amount-input").value = 50000;
}

async function doStep() {
  if (!selectedAction || stepping) return;
  stepping = true;
  $("btn-step").disabled = true;

  const amount = selectedAction.amount ? Number($("amount-input").value || 0) : 0;
  const res = await fetch(`${API}/step?session_id=${SESSION}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action_type: selectedAction.id, amount }),
  });
  if (!res.ok) {
    toast(`Step failed: ${(await res.json()).detail}`, true);
    stepping = false;
    $("btn-step").disabled = false;
    return;
  }
  const result = await res.json();
  const info = result.info;
  prevNetWorth = obs.net_worth;
  obs = result.observation;

  netWorthHistory.push(obs.net_worth);
  rewards.push(result.reward);

  // log: action result
  logLine(
    `<strong>M${obs.month - 1}</strong> · ${info.action_message} Reward <strong>${result.reward.toFixed(2)}</strong>`,
    info.action_ok ? "" : "bad"
  );
  if (!info.action_ok) toast(info.action_message, true);

  // life event
  const evText = EVENT_TEXT[info.event];
  const banner = $("event-banner");
  if (evText) {
    banner.hidden = false;
    banner.textContent = evText;
    logLine(evText, "event");
  } else {
    banner.hidden = true;
  }

  render();

  if (result.done) {
    setTimeout(showResults, 700);
  } else {
    $("btn-step").disabled = false;
  }
  stepping = false;
}

/* ------------------------------------------------------------ rendering */
function render() {
  // KPIs
  $("kpi-networth").textContent = fmtFull(obs.net_worth);
  const deltaEl = $("kpi-networth-delta");
  if (prevNetWorth !== null) {
    const d = obs.net_worth - prevNetWorth;
    deltaEl.textContent = `${d >= 0 ? "▲" : "▼"} ${fmtMoney(Math.abs(d))} this month`;
    deltaEl.className = `kpi-delta ${d >= 0 ? "pos" : "neg"}`;
  } else {
    deltaEl.textContent = "";
  }

  $("kpi-savings").textContent = fmtFull(obs.savings);
  $("kpi-cashflow").textContent =
    `income ${fmtMoney(obs.income)}/mo · spend ${fmtMoney(obs.fixed_expenses + obs.variable_expenses)}/mo`;

  const efTarget = 3 * obs.fixed_expenses;
  $("kpi-ef").textContent = fmtFull(obs.emergency_fund);
  $("kpi-ef-bar").style.width = `${Math.min(100, (obs.emergency_fund / efTarget) * 100)}%`;
  $("kpi-ef-target").textContent = `target ${fmtMoney(efTarget)}`;

  const totalDebt = obs.debt.credit_card + obs.debt.personal_loan;
  $("kpi-debt").textContent = fmtFull(totalDebt);
  $("kpi-debt-split").textContent =
    `card ${fmtMoney(obs.debt.credit_card)} · loan ${fmtMoney(obs.debt.personal_loan)}`;

  // credit gauge
  $("kpi-credit").textContent = Math.round(obs.credit_score);
  const frac = (obs.credit_score - 300) / 550;
  const fill = $("gauge-fill");
  fill.style.strokeDashoffset = 157 * (1 - frac);
  fill.style.stroke = obs.credit_score >= 700 ? "#34d399" : obs.credit_score >= 600 ? "#fbbf24" : "#f87171";

  // regime
  const regime = $("hud-regime");
  regime.textContent = { bull: "▲ Bull", bear: "▼ Bear", sideways: "◆ Sideways" }[obs.market_regime];
  regime.className = `regime-badge regime-${obs.market_regime}`;

  // month track
  const track = $("month-track");
  track.innerHTML = "";
  for (let m = 1; m <= 6; m++) {
    const dot = document.createElement("div");
    dot.className = "month-dot" + (m < obs.month ? " done" : m === obs.month ? " current" : "");
    dot.title = `Month ${m}`;
    track.appendChild(dot);
  }
  $("action-month-label").textContent = obs.month <= 6 ? `Month ${obs.month} of 6` : "Done";

  // cash flow panel
  $("cf-income").textContent = `+${fmtMoney(obs.income)}`;
  $("cf-fixed").textContent = `−${fmtMoney(obs.fixed_expenses)}`;
  $("cf-variable").textContent = `−${fmtMoney(obs.variable_expenses)}`;
  const net = obs.income - obs.fixed_expenses - obs.variable_expenses;
  const cfNet = $("cf-net");
  cfNet.textContent = fmtMoney(net);
  cfNet.className = net >= 0 ? "pos" : "neg";

  // reward chips
  const chips = $("reward-chips");
  chips.innerHTML = "";
  rewards.forEach((r) => {
    const c = document.createElement("span");
    c.className = `reward-chip ${r >= 0 ? "pos" : "neg"}`;
    c.textContent = (r >= 0 ? "+" : "") + r.toFixed(2);
    chips.appendChild(c);
  });

  updateCharts();
}

function logLine(html, cls = "") {
  const li = document.createElement("li");
  if (cls) li.className = cls;
  li.innerHTML = html;
  const log = $("activity-log");
  log.prepend(li);
}

function toast(msg, isError = false) {
  const el = document.createElement("div");
  el.className = `toast${isError ? " error" : ""}`;
  el.textContent = msg;
  $("toast-stack").appendChild(el);
  setTimeout(() => el.remove(), 4200);
}

/* ------------------------------------------------------------ charts */
const CHART_DEFAULTS = {
  color: "#8b93a7",
  font: { family: "Inter" },
};

function initCharts() {
  Chart.defaults.color = CHART_DEFAULTS.color;
  Chart.defaults.font.family = "Inter";
  Chart.defaults.borderColor = "rgba(255,255,255,0.06)";

  if (chartNet) chartNet.destroy();
  if (chartPortfolio) chartPortfolio.destroy();

  const ctx = $("chart-networth").getContext("2d");
  const grad = ctx.createLinearGradient(0, 0, 0, 260);
  grad.addColorStop(0, "rgba(52, 211, 153, 0.35)");
  grad.addColorStop(1, "rgba(52, 211, 153, 0)");

  chartNet = new Chart(ctx, {
    type: "line",
    data: {
      labels: ["Start"],
      datasets: [{
        data: [...netWorthHistory],
        borderColor: "#34d399",
        backgroundColor: grad,
        fill: true,
        tension: 0.35,
        pointRadius: 4,
        pointBackgroundColor: "#34d399",
      }],
    },
    options: {
      maintainAspectRatio: false,
      plugins: { legend: { display: false },
                 tooltip: { callbacks: { label: (c) => fmtFull(c.parsed.y) } } },
      scales: {
        y: { ticks: { callback: (v) => fmtMoney(v) }, grid: { color: "rgba(255,255,255,0.05)" } },
        x: { grid: { display: false } },
      },
    },
  });

  chartPortfolio = new Chart($("chart-portfolio"), {
    type: "doughnut",
    data: {
      labels: ["Savings", "Emergency", "Stocks", "Crypto", "Bonds", "FD", "Mutual funds", "Commodities", "Real estate"],
      datasets: [{
        data: [],
        backgroundColor: ["#34d399", "#2dd4bf", "#38bdf8", "#f472b6", "#818cf8",
                          "#a78bfa", "#60a5fa", "#fbbf24", "#fb923c"],
        borderColor: "#0c101a",
        borderWidth: 3,
      }],
    },
    options: {
      maintainAspectRatio: false,
      cutout: "62%",
      plugins: {
        legend: { position: "right", labels: { boxWidth: 10, font: { size: 11 } } },
        tooltip: { callbacks: { label: (c) => ` ${c.label}: ${fmtFull(c.parsed)}` } },
      },
    },
  });
}

function updateCharts() {
  chartNet.data.labels = netWorthHistory.map((_, i) => (i === 0 ? "Start" : `M${i}`));
  chartNet.data.datasets[0].data = [...netWorthHistory];
  chartNet.update();

  const inv = obs.investments;
  chartPortfolio.data.datasets[0].data = [
    Math.max(0, obs.savings), Math.max(0, obs.emergency_fund),
    inv.stocks, inv.crypto, inv.bonds, inv.fd,
    inv.mutual_funds, inv.commodities, inv.real_estate,
  ].map((v) => Math.round(v));
  chartPortfolio.update();
}

/* ------------------------------------------------------------ results */
const GRADE_LETTERS = [[0.9, "S — LEGENDARY"], [0.8, "A — EXCELLENT"], [0.65, "B — SOLID"],
                       [0.5, "C — SURVIVED"], [0.3, "D — ROUGH"], [0, "F — WIPED OUT"]];

async function showResults() {
  let data;
  try {
    const res = await fetch(`${API}/grade?session_id=${SESSION}`, { method: "POST" });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      toast(`Grading failed: ${err.detail || res.statusText}`, true);
      $("btn-step").disabled = false;
      stepping = false;
      return;
    }
    data = await res.json();
  } catch (e) {
    toast("Could not reach server for grading.", true);
    $("btn-step").disabled = false;
    stepping = false;
    return;
  }

  $("results-modal").hidden = false;
  $("result-title").textContent = `${selectedTask.name} — Complete`;

  const score = data.score;
  $("ring-score").textContent = score.toFixed(2);
  $("ring-grade").textContent = GRADE_LETTERS.find(([min]) => score >= min)[1];
  const ring = $("ring-fill");
  const C = 2 * Math.PI * 84;
  requestAnimationFrame(() => {
    ring.style.strokeDashoffset = C * (1 - score);
    ring.style.stroke = score >= 0.65 ? "#34d399" : score >= 0.4 ? "#fbbf24" : "#f87171";
  });

  const bk = $("score-breakdown");
  bk.innerHTML = "";
  for (const c of data.components) {
    const row = document.createElement("div");
    row.className = "bk-row";
    row.innerHTML = `
      <div class="bk-row-head"><span>${c.label}</span><strong>${c.score.toFixed(2)} / ${c.max}</strong></div>
      <div class="bk-bar"><div class="bk-bar-fill" style="width:0%"></div></div>`;
    bk.appendChild(row);
    requestAnimationFrame(() =>
      requestAnimationFrame(() => {
        row.querySelector(".bk-bar-fill").style.width = `${(c.score / c.max) * 100}%`;
      })
    );
  }

  const startNet = netWorthHistory[0];
  const endNet = netWorthHistory[netWorthHistory.length - 1];
  const totalDebt = obs.debt.credit_card + obs.debt.personal_loan;
  $("result-stats").innerHTML = `
    <div>Net worth <strong>${fmtFull(endNet)}</strong></div>
    <div>Change <strong>${endNet - startNet >= 0 ? "+" : ""}${fmtMoney(endNet - startNet)}</strong></div>
    <div>Remaining debt <strong>${fmtFull(totalDebt)}</strong></div>
    <div>Credit score <strong>${Math.round(obs.credit_score)}</strong></div>`;
}

/* ------------------------------------------------------------ wiring */
$("btn-start").addEventListener("click", startEpisode);
$("btn-step").addEventListener("click", doStep);
function goToStartScreen() {
  $("results-modal").hidden = true;
  $("screen-game").classList.remove("active");
  $("screen-start").classList.add("active");
  stepping = false;
  selectedAction = null;
  $("btn-step").disabled = true;
}

$("btn-restart").addEventListener("click", goToStartScreen);
$("btn-again").addEventListener("click", goToStartScreen);
function quickAmountBase() {
  if (!obs) return 0;
  if (!selectedAction) return obs.savings;
  const id = selectedAction.id;
  if (id.startsWith("sell_")) return obs.investments[id.slice(5)] ?? 0;
  if (id === "withdraw_emergency_fund") return obs.emergency_fund;
  if (id === "pay_credit_card") return Math.min(obs.savings, obs.debt.credit_card);
  if (id === "pay_personal_loan") return Math.min(obs.savings, obs.debt.personal_loan);
  return obs.savings;
}

document.querySelectorAll(".quick-amounts button").forEach((b) =>
  b.addEventListener("click", () => {
    $("amount-input").value = Math.floor(quickAmountBase() * Number(b.dataset.pct));
  })
);

function initUI() {
  $("results-modal").hidden = true;
  $("event-banner").hidden = true;
  $("screen-start").classList.add("active");
  $("screen-game").classList.remove("active");
}

buildActionPanel();
initUI();
loadTasks().catch(() => toast("Could not reach the FinAgent server. Is it running on port 7860?", true));
