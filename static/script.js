// ─── State ────────────────────────────────────────────────────────────────────
const ratings = {
  director:  6.0,
  lead:      6.0,
  support1:  6.0,
  support2:  6.0,
};

// ─── Trend Selector ───────────────────────────────────────────────────────────
document.querySelectorAll('.trend-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.trend-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('genre_trend').value = btn.dataset.value;
  });
});

// ─── TMDB Person Lookup ───────────────────────────────────────────────────────
async function lookupPerson(role) {
  const nameEl     = document.getElementById(roleToId(role, 'name'));
  const spinnerId  = `${role}-spinner`;
  const resultId   = `${role}-result`;
  const name       = nameEl.value.trim();

  if (!name) return;

  const year = parseInt(document.getElementById('release_year').value) || 2024;

  // UI: loading
  const spinner = document.getElementById(spinnerId);
  const btnText = spinner.previousElementSibling;
  spinner.classList.remove('hidden');
  btnText.classList.add('hidden');

  try {
    const resp = await fetch('/api/lookup-person', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name,
        department: role === 'director' ? 'Directing' : 'Acting',
        before_year: year
      })
    });

    const data = await resp.json();

    // Store rating
    ratings[role] = data.rating;
    if (role === 'director') document.getElementById('director_rating').value = data.rating;
    if (role === 'lead')     document.getElementById('lead_rating').value     = data.rating;
    if (role === 'support1') document.getElementById('support1_rating').value = data.rating;
    if (role === 'support2') document.getElementById('support2_rating').value = data.rating;

    // Show result card
    const resultEl = document.getElementById(resultId);
    document.getElementById(`${role}-found-name`).textContent = data.found_name;

    const dataStatus = data.data_found
      ? `AVG RATING PRE-${year}: ${data.rating} / 10`
      : `NO PAST DATA — DEFAULT: ${data.rating} / 10`;
    document.getElementById(`${role}-rating-display`).textContent = dataStatus;

    const avatarEl = document.getElementById(`${role}-avatar`);
    if (data.profile_path) {
      avatarEl.innerHTML = `<img src="${data.profile_path}" alt="${data.found_name}" />`;
    } else {
      avatarEl.innerHTML = `<div style="display:flex;align-items:center;justify-content:center;height:100%;font-size:11px;color:var(--text-muted)">?</div>`;
    }

    resultEl.classList.remove('hidden');

  } catch (err) {
    console.error(err);
  } finally {
    spinner.classList.add('hidden');
    btnText.classList.remove('hidden');
  }
}

function roleToId(role, type) {
  const map = {
    director:  { name: 'director_name' },
    lead:      { name: 'lead_name' },
    support1:  { name: 'support1_name' },
    support2:  { name: 'support2_name' },
  };
  return map[role][type];
}

// ─── Prediction ───────────────────────────────────────────────────────────────
async function runPrediction() {
  const btn = document.getElementById('predict-btn');
  const errorEl = document.getElementById('predict-error');
  errorEl.classList.add('hidden');

  // Validate budget
  const budgetRaw = document.getElementById('budget').value.replace(/,/g, '').trim();
  if (!budgetRaw || isNaN(parseFloat(budgetRaw))) {
    showError('Please enter a valid budget.');
    return;
  }

  const support1 = parseFloat(document.getElementById('support1_rating').value) || 6.0;
  const support2 = parseFloat(document.getElementById('support2_rating').value) || 6.0;
  const sidecasstAvg = ((support1 + support2) / 2).toFixed(2);

  const payload = {
    budget:          parseFloat(budgetRaw),
    release_year:    parseInt(document.getElementById('release_year').value) || 2025,
    release_month:   parseInt(document.getElementById('release_month').value),
    num_genres:      parseInt(document.getElementById('num_genres').value) || 1,
    director_rating: parseFloat(document.getElementById('director_rating').value) || 6.0,
    lead_actor_rating: parseFloat(document.getElementById('lead_rating').value) || 6.0,
    sidecast_rating: parseFloat(sidecasstAvg),
    genre_trend:     parseInt(document.getElementById('genre_trend').value),
    primary_genre:   document.getElementById('primary_genre').value,
    is_action:       document.getElementById('is_action').checked ? 1 : 0,
    is_comedy:       document.getElementById('is_comedy').checked ? 1 : 0,
    is_drama:        document.getElementById('is_drama').checked ? 1 : 0,
    is_horror:       document.getElementById('is_horror').checked ? 1 : 0,
    is_sciencefiction: document.getElementById('is_sciencefiction').checked ? 1 : 0,
    is_animation:    document.getElementById('is_animation').checked ? 1 : 0,
    is_romance:      document.getElementById('is_romance').checked ? 1 : 0,
    is_thriller:     document.getElementById('is_thriller').checked ? 1 : 0,
  };

  // UI: Show loading
  setUIState('loading');
  btn.classList.add('loading');
  btn.querySelector('.predict-btn-text').textContent = 'ANALYSING...';

  try {
    const resp = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const data = await resp.json();
    if (data.error) throw new Error(data.error);

    setUIState('results');
    renderResults(data, payload.budget);

  } catch (err) {
    setUIState('idle');
    showError(`Prediction failed: ${err.message}`);
  } finally {
    btn.classList.remove('loading');
    btn.querySelector('.predict-btn-text').textContent = 'RUN ANALYSIS';
  }
}

// ─── Render Results ───────────────────────────────────────────────────────────
function renderResults(data, budget) {
  // Verdict
  const verdictEl = document.getElementById('verdict-value');
  verdictEl.textContent = data.verdict.toUpperCase();
  verdictEl.className = 'verdict-value ' + data.verdict.toLowerCase();

  // Confidence bar
  setTimeout(() => {
    document.getElementById('confidence-bar').style.width = `${data.confidence}%`;
    document.getElementById('confidence-pct').textContent = `${data.confidence}%`;
  }, 150);

  // Probability bars
  const probContainer = document.getElementById('prob-bars');
  probContainer.innerHTML = '';
  const verdictOrder = ['Hit', 'Average', 'Flop'];
  verdictOrder.forEach((label, i) => {
    const pct = data.all_probs[label] || 0;
    const row = document.createElement('div');
    row.className = 'prob-row';
    row.innerHTML = `
      <span class="prob-row-label">${label.toUpperCase()}</span>
      <div class="prob-row-bar-outer">
        <div class="prob-row-bar-inner ${label.toLowerCase()}" style="width:0%" data-target="${pct}"></div>
      </div>
      <span class="prob-row-pct">${pct.toFixed(1)}%</span>
    `;
    probContainer.appendChild(row);
    setTimeout(() => {
      row.querySelector('.prob-row-bar-inner').style.width = `${pct}%`;
    }, 200 + i * 80);
  });

  // Revenue
  animateCount('revenue-value', 0, data.revenue, 1400, v => formatRevenue(v));
  const roi = (data.revenue / budget).toFixed(2);
  const roiColor = roi >= 2.5 ? 'var(--green)' : roi >= 1.5 ? 'var(--amber)' : 'var(--red)';
  document.getElementById('revenue-roi').innerHTML = `ROI MULTIPLE: <span style="color:${roiColor};font-weight:700">${roi}×</span>`;

  // Revenue Arc (max scale = $1B)
  const arcPct = Math.min(data.revenue / 1_000_000_000, 1);
  const arcLen = 125;
  setTimeout(() => {
    document.getElementById('revenue-arc').setAttribute('stroke-dasharray', `${arcPct * arcLen} ${arcLen}`);
  }, 300);
  document.getElementById('revenue-card').classList.add('revealed');

  // Rating
  animateCount('rating-value', 0, data.imdb_rating, 1200, v => v.toFixed(1) + ' / 10');
  renderStars(data.imdb_rating);
  document.getElementById('rating-card').classList.add('revealed');
}

// ─── Animate Number Count ─────────────────────────────────────────────────────
function animateCount(elId, from, to, duration, formatter) {
  const el = document.getElementById(elId);
  const start = performance.now();
  function update(ts) {
    const elapsed = ts - start;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 4);
    el.textContent = formatter(from + (to - from) * eased);
    if (progress < 1) requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
}

function formatRevenue(v) {
  if (v >= 1_000_000_000) return `$${(v / 1_000_000_000).toFixed(2)}B`;
  if (v >= 1_000_000)     return `$${(v / 1_000_000).toFixed(1)}M`;
  return `$${Math.round(v).toLocaleString()}`;
}

// ─── Rating Stars ─────────────────────────────────────────────────────────────
function renderStars(rating) {
  const container = document.getElementById('rating-stars');
  container.innerHTML = '';
  for (let i = 1; i <= 10; i++) {
    const star = document.createElement('span');
    star.className = 'star';
    star.textContent = '◆';
    container.appendChild(star);
    if (i <= Math.round(rating)) {
      setTimeout(() => star.classList.add('lit'), i * 60 + 400);
    }
  }
}

// ─── UI State Management ──────────────────────────────────────────────────────
function setUIState(state) {
  document.getElementById('idle-state').classList.add('hidden');
  document.getElementById('loading-state').classList.add('hidden');
  document.getElementById('results-content').classList.add('hidden');

  if (state === 'idle')    document.getElementById('idle-state').classList.remove('hidden');
  if (state === 'loading') document.getElementById('loading-state').classList.remove('hidden');
  if (state === 'results') document.getElementById('results-content').classList.remove('hidden');
}

function resetResults() {
  setUIState('idle');
  document.getElementById('confidence-bar').style.width = '0%';
}

function showError(msg) {
  const el = document.getElementById('predict-error');
  el.textContent = msg;
  el.classList.remove('hidden');
}

// ─── Allow Enter key on talent inputs to trigger lookup ───────────────────────
document.getElementById('director_name').addEventListener('keydown', e => {
  if (e.key === 'Enter') lookupPerson('director');
});
document.getElementById('lead_name').addEventListener('keydown', e => {
  if (e.key === 'Enter') lookupPerson('lead');
});
document.getElementById('support1_name').addEventListener('keydown', e => {
  if (e.key === 'Enter') lookupPerson('support1');
});
document.getElementById('support2_name').addEventListener('keydown', e => {
  if (e.key === 'Enter') lookupPerson('support2');
});
