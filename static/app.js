// Professional Energy Management System Dashboard
// Real-time monitoring and control interface

// Global state
let charts = {
    episode: null,
    comparison: null,
    training: null,
    dataset: null,
    algorithm: null,
    ablation: null,
    pareto: null
};

let currentMetrics = {
    peak: null,
    cost: null,
    comfort: null,
    reward: null
};

function showChartMessage(canvasId, message) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const parent = canvas.parentElement;
    if (!parent) return;

    canvas.style.display = 'none';

    let msgEl = parent.querySelector('.chart-empty-message');
    if (!msgEl) {
        msgEl = document.createElement('div');
        msgEl.className = 'chart-empty-message';
        msgEl.style.display = 'flex';
        msgEl.style.alignItems = 'center';
        msgEl.style.justifyContent = 'center';
        msgEl.style.minHeight = '280px';
        msgEl.style.padding = '1rem';
        msgEl.style.textAlign = 'center';
        msgEl.style.color = 'var(--text-secondary)';
        msgEl.style.fontSize = '0.95rem';
        msgEl.style.border = '1px dashed var(--border)';
        msgEl.style.borderRadius = '10px';
        parent.appendChild(msgEl);
    }

    msgEl.textContent = message;
}

function clearChartMessage(canvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const parent = canvas.parentElement;
    if (!parent) return;

    const msgEl = parent.querySelector('.chart-empty-message');
    if (msgEl) {
        msgEl.remove();
    }
    canvas.style.display = '';
}

// Initialize dashboard on page load
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
});

function initializeDashboard() {
    updateSystemTime();
    setInterval(updateSystemTime, 1000);
    
    loadDatasetInfo();
    loadSummary();
    loadTrainingStats();
    loadDatasetPreview();
    loadAlgorithmComparison();
    loadAblationSummary();
    loadParetoData();
    loadFinalSummary();
    
    setupEventListeners();
    addLogEntry('System initialized. Dashboard ready.');
}

// Update system time display
function updateSystemTime() {
    const now = new Date();
    const timeStr = now.toLocaleTimeString('en-US', { hour12: false });
    const timeEl = document.getElementById('currentTime');
    if (timeEl) {
        timeEl.textContent = timeStr;
    }
}

// Setup event listeners
function setupEventListeners() {
    // Run simulation button
    const runBtn = document.getElementById('runSimulationBtn');
    if (runBtn) {
        runBtn.addEventListener('click', handleRunSimulation);
    }
    
    // Export button
    const exportBtn = document.getElementById('exportDataBtn');
    if (exportBtn) {
        exportBtn.addEventListener('click', handleExportData);
    }
    
    // Modal close
    const modalClose = document.getElementById('modalClose');
    if (modalClose) {
        modalClose.addEventListener('click', closeImageModal);
    }
    
    // Close modal on Escape
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            closeImageModal();
        }
    });
}

// Handle simulation run
async function handleRunSimulation() {
    const btn = document.getElementById('runSimulationBtn');
    const policySelect = document.getElementById('policySelect');
    const modeSelect = document.getElementById('modeSelect');
    const infoBadge = document.querySelector('.info-badge');
    
    if (!btn || !policySelect || !modeSelect) return;
    
    const policy = policySelect.value;
    const mode = modeSelect.value;
    
    btn.disabled = true;
    btn.innerHTML = '<span class="btn-icon">⏳</span> Running...';
    if (infoBadge) {
        infoBadge.textContent = 'Running...';
        infoBadge.style.color = 'var(--warning)';
    }
    
    addLogEntry(`Starting ${policy === 'rl' ? 'RL Agent' : 'Baseline'} simulation with ${mode === 'real' ? 'Real Dataset' : 'Synthetic'} data...`);
    
    try {
        const response = await fetch(`/api/run_simulation?policy=${encodeURIComponent(policy)}&mode=${encodeURIComponent(mode)}`);
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        if (data.timesteps && data.timesteps.length > 0) {
            createEpisodeChart(data.timesteps);
            updateLiveMetrics(data.metrics, policy, mode);
            updateRealTimeMetrics(data.metrics);
            addLogEntry(`Simulation completed. Peak: ${data.metrics.peak_demand_kw.toFixed(2)} kW, Cost: ₹${data.metrics.total_cost.toFixed(2)}`);
        }
    } catch (error) {
        console.error('Simulation error:', error);
        addLogEntry(`Error: ${error.message}`, 'error');
        if (infoBadge) {
            infoBadge.textContent = 'Error';
            infoBadge.style.color = 'var(--danger)';
        }
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<span class="btn-icon">▶</span> Run Simulation';
        if (infoBadge) {
            infoBadge.textContent = 'Ready';
            infoBadge.style.color = 'var(--text-secondary)';
        }
    }
}

// Update real-time metrics cards
function updateRealTimeMetrics(metrics) {
    if (!metrics) return;
    
    currentMetrics.peak = metrics.peak_demand_kw;
    currentMetrics.cost = metrics.total_cost;
    currentMetrics.comfort = metrics.min_comfort;
    currentMetrics.reward = metrics.total_reward;
    
    updateMetricCard('metricPeak', metrics.peak_demand_kw.toFixed(3), 'kW');
    updateMetricCard('metricCost', `₹${metrics.total_cost.toFixed(2)}`, 'INR');
    updateMetricCard('metricComfort', metrics.min_comfort.toFixed(3), '0-1 Scale');
    updateMetricCard('metricReward', metrics.total_reward.toFixed(2), 'Cumulative');
}

function updateMetricCard(id, value, unit) {
    const valueEl = document.getElementById(id);
    if (valueEl) {
        valueEl.textContent = value;
    }
}

// Create episode chart
function createEpisodeChart(data) {
    const ctx = document.getElementById('episodeChart');
    if (!ctx) return;
    
    if (charts.episode) {
        charts.episode.destroy();
    }
    
    const hours = data.map(d => d.hour);
    const pv = data.map(d => parseFloat(d.pv));
    const gridImport = data.map(d => parseFloat(d.grid_import));
    const soc = data.map(d => parseFloat(d.soc) * 100);
    const comfort = data.map(d => parseFloat(d.comfort) * 100);
    
    charts.episode = new Chart(ctx, {
        type: 'line',
        data: {
            labels: hours,
            datasets: [
                {
                    label: 'PV Generation (kW)',
                    data: pv,
                    borderColor: 'rgb(16, 185, 129)',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    borderWidth: 2,
                    tension: 0.3
                },
                {
                    label: 'Grid Import (kW)',
                    data: gridImport,
                    borderColor: 'rgb(37, 99, 235)',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    borderWidth: 2,
                    tension: 0.3
                },
                {
                    label: 'Battery SOC (%)',
                    data: soc,
                    borderColor: 'rgb(239, 68, 68)',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    borderWidth: 2,
                    tension: 0.3,
                    yAxisID: 'y1'
                },
                {
                    label: 'Comfort (%)',
                    data: comfort,
                    borderColor: 'rgb(168, 85, 247)',
                    backgroundColor: 'rgba(168, 85, 247, 0.1)',
                    borderWidth: 2,
                    tension: 0.3,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#f1f5f9' }
                },
                title: {
                    display: true,
                    text: '24-Hour Simulation: Real-Time Metrics',
                    color: '#f1f5f9',
                    font: { size: 16, weight: 'bold' }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Hour of Day',
                        color: '#94a3b8'
                    },
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(148, 163, 184, 0.1)' }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Power (kW)',
                        color: '#94a3b8'
                    },
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(148, 163, 184, 0.1)' }
                },
                y1: {
                    type: 'linear',
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Percentage (%)',
                        color: '#94a3b8'
                    },
                    ticks: { color: '#94a3b8' },
                    grid: { drawOnChartArea: false }
                }
            }
        }
    });
}

// Update live simulation metrics summary
function updateLiveMetrics(metrics, policy, mode) {
    const container = document.getElementById('liveSimulationMetrics');
    if (!container || !metrics) return;
    
    const policyLabel = policy === 'baseline' ? 'Baseline Controller' : 'RL Agent (PPO)';
    const modeLabel = mode === 'real' ? 'Real Dataset' : 'Synthetic Environment';
    
    container.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
            <div><strong>${policyLabel}</strong> on <strong>${modeLabel}</strong></div>
            <div style="display: flex; gap: 2rem;">
                <span>Peak: <strong>${metrics.peak_demand_kw.toFixed(2)} kW</strong></span>
                <span>Cost: <strong>₹${metrics.total_cost.toFixed(2)}</strong></span>
                <span>Comfort: <strong>${metrics.min_comfort.toFixed(3)}</strong></span>
                <span>Reward: <strong>${metrics.total_reward.toFixed(2)}</strong></span>
            </div>
        </div>
    `;
}

// Load summary and create comparison chart (cost from CSV is in USD, we convert to INR for display)
async function loadSummary() {
    try {
        const [summaryRes, infoRes] = await Promise.all([fetch('/api/summary'), fetch('/api/project_info')]);
        const data = await summaryRes.json();
        const info = await infoRes.json();
        const usdToInr = info.usd_to_inr || 83;
        
        if (data.error || !data.length) {
            showChartMessage('comparisonChart', 'Baseline vs RL summary is missing. Generate results/logs/summary.csv to view this chart.');
            return;
        }
        
        const rlData = data.find(d => d.Policy === 'RL_PPO');
        const baselineData = data.find(d => d.Policy === 'Baseline');
        
        if (rlData && baselineData) {
            createComparisonChart(rlData, baselineData, usdToInr);
            return;
        }
        showChartMessage('comparisonChart', 'Summary exists but expected Baseline and RL_PPO rows were not found.');
    } catch (error) {
        console.error('Error loading summary:', error);
        showChartMessage('comparisonChart', 'Could not load comparison chart due to a data loading error.');
    }
}

// Create comparison chart (cost values converted to INR via usdToInr)
function createComparisonChart(rlData, baselineData, usdToInr) {
    usdToInr = usdToInr || 83;
    const ctx = document.getElementById('comparisonChart');
    if (!ctx) return;
    clearChartMessage('comparisonChart');
    
    if (charts.comparison) {
        charts.comparison.destroy();
    }
    
    const rlCostInr = (rlData.Mean_Total_Cost || 0) * usdToInr;
    const baselineCostInr = (baselineData.Mean_Total_Cost || 0) * usdToInr;
    
    charts.comparison = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Reward', 'Peak Demand (kW)', 'Total Cost (₹)', 'Min Comfort'],
            datasets: [
                {
                    label: 'RL Agent',
                    data: [
                        rlData.Mean_Reward,
                        rlData.Mean_Peak_Demand_kW,
                        rlCostInr,
                        rlData.Mean_Min_Comfort * 100
                    ],
                    backgroundColor: 'rgba(16, 185, 129, 0.7)',
                    borderColor: 'rgb(16, 185, 129)',
                    borderWidth: 2
                },
                {
                    label: 'Baseline',
                    data: [
                        baselineData.Mean_Reward,
                        baselineData.Mean_Peak_Demand_kW,
                        baselineCostInr,
                        baselineData.Mean_Min_Comfort * 100
                    ],
                    backgroundColor: 'rgba(245, 158, 11, 0.7)',
                    borderColor: 'rgb(245, 158, 11)',
                    borderWidth: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#f1f5f9' }
                },
                title: {
                    display: true,
                    text: 'RL Agent vs Baseline Controller Performance Comparison',
                    color: '#f1f5f9',
                    font: { size: 16, weight: 'bold' }
                }
            },
            scales: {
                y: {
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(148, 163, 184, 0.1)' }
                },
                x: {
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(148, 163, 184, 0.1)' }
                }
            }
        }
    });
}

// Load training stats
async function loadTrainingStats() {
    try {
        const response = await fetch('/api/training_stats');
        const data = await response.json();
        
        if (data.error || !data.rewards || data.rewards.length === 0) {
            showChartMessage('trainingChart', 'Training stats are unavailable. Run PPO training to generate monitor.csv.');
            return;
        }
        
        createTrainingChart(data);
    } catch (error) {
        console.error('Error loading training stats:', error);
        showChartMessage('trainingChart', 'Could not load training stats due to a data loading error.');
    }
}

// Create training chart
function createTrainingChart(data) {
    const ctx = document.getElementById('trainingChart');
    if (!ctx) return;
    clearChartMessage('trainingChart');
    
    if (charts.training) {
        charts.training.destroy();
    }
    
    const windowSize = Math.max(1, Math.floor(data.rewards.length / 50));
    const smoothedRewards = [];
    for (let i = 0; i < data.rewards.length; i++) {
        const start = Math.max(0, i - windowSize);
        const end = Math.min(data.rewards.length, i + windowSize);
        const slice = data.rewards.slice(start, end);
        smoothedRewards.push(slice.reduce((a, b) => a + b, 0) / slice.length);
    }
    
    charts.training = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({length: data.rewards.length}, (_, i) => `Episode ${i + 1}`),
            datasets: [
                {
                    label: 'Episode Reward',
                    data: data.rewards,
                    borderColor: 'rgba(37, 99, 235, 0.5)',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    borderWidth: 1,
                    pointRadius: 0
                },
                {
                    label: 'Smoothed Reward',
                    data: smoothedRewards,
                    borderColor: 'rgb(37, 99, 235)',
                    backgroundColor: 'rgba(37, 99, 235, 0.2)',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#f1f5f9' }
                },
                title: {
                    display: true,
                    text: 'RL Training Progress: Episode Rewards Over Time',
                    color: '#f1f5f9',
                    font: { size: 16, weight: 'bold' }
                }
            },
            scales: {
                y: {
                    title: {
                        display: true,
                        text: 'Reward',
                        color: '#94a3b8'
                    },
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(148, 163, 184, 0.1)' }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Episode',
                        color: '#94a3b8'
                    },
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(148, 163, 184, 0.1)' }
                }
            }
        }
    });
}

// Load dataset info
async function loadDatasetInfo() {
    try {
        const response = await fetch('/api/dataset_info');
        const data = await response.json();
        
        const container = document.getElementById('datasetStatus');
        if (!container) return;
        
        if (!data.exists) {
            container.innerHTML = '<div style="color: var(--warning);">Dataset not found</div>';
            return;
        }
        
        container.innerHTML = `
            <div style="margin-bottom: 0.5rem;">
                <strong style="color: var(--success);">✓ Available</strong>
            </div>
            <div style="font-size: 0.75rem; color: var(--text-secondary);">
                ${data.num_rows.toLocaleString()} records<br>
                ${data.date_min ? data.date_min.split(' ')[0] : 'N/A'} to ${data.date_max ? data.date_max.split(' ')[0] : 'N/A'}
            </div>
        `;
    } catch (error) {
        console.error('Error loading dataset info:', error);
    }
}

// Load dataset preview
async function loadDatasetPreview() {
    try {
        const response = await fetch('/api/dataset_preview');
        const data = await response.json();
        
        if (data.error || !data.exists) {
            showChartMessage('realDatasetChart', `Real dataset insights are unavailable. ${data.error || 'Dataset is missing.'}`);
            return;
        }
        
        createDatasetChart(data);
    } catch (error) {
        console.error('Error loading dataset preview:', error);
        showChartMessage('realDatasetChart', 'Could not load real dataset chart due to a data loading error.');
    }
}

// Create dataset chart
function createDatasetChart(data) {
    const ctx = document.getElementById('realDatasetChart');
    if (!ctx) return;
    clearChartMessage('realDatasetChart');
    
    if (charts.dataset) {
        charts.dataset.destroy();
    }
    
    charts.dataset = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.hours,
            datasets: [
                {
                    label: 'Average PV Generation (kW)',
                    data: data.pv,
                    borderColor: 'rgb(16, 185, 129)',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    borderWidth: 2,
                    tension: 0.3
                },
                {
                    label: 'Average Load (kW)',
                    data: data.load,
                    borderColor: 'rgb(239, 68, 68)',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    borderWidth: 2,
                    tension: 0.3
                },
                {
                    label: 'Average Price ($/kWh)',
                    data: data.price,
                    borderColor: 'rgb(59, 130, 246)',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 2,
                    tension: 0.3,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#f1f5f9' }
                },
                title: {
                    display: true,
                    text: 'Real-World Dataset: Average 24-Hour Profile',
                    color: '#f1f5f9',
                    font: { size: 16, weight: 'bold' }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Hour of Day',
                        color: '#94a3b8'
                    },
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(148, 163, 184, 0.1)' }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Power (kW)',
                        color: '#94a3b8'
                    },
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(148, 163, 184, 0.1)' }
                },
                y1: {
                    type: 'linear',
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Price ($/kWh)',
                        color: '#94a3b8'
                    },
                    ticks: { color: '#94a3b8' },
                    grid: { drawOnChartArea: false }
                }
            }
        }
    });
}

async function loadAlgorithmComparison() {
    try {
        const response = await fetch('/api/algorithm_comparison');
        const data = await response.json();
        const ctx = document.getElementById('algorithmChart');
        if (!ctx) return;
        if (!data.available) {
            addLogEntry(`Algorithm comparison unavailable. ${data.hint || ''}`);
            showChartMessage('algorithmChart', `Algorithm comparison data is not available yet. ${data.hint || 'Run the comparison experiment to populate this chart.'}`);
            return;
        }
        const rows = data.rows || [];
        if (!rows.length) {
            showChartMessage('algorithmChart', 'Algorithm comparison file is present but contains no rows.');
            return;
        }
        createAlgorithmChart(rows);
    } catch (error) {
        console.error('Error loading algorithm comparison:', error);
    }
}

function createAlgorithmChart(rows) {
    const ctx = document.getElementById('algorithmChart');
    if (!ctx) return;
    clearChartMessage('algorithmChart');
    if (charts.algorithm) charts.algorithm.destroy();

    const labels = rows.map(r => r.algorithm);
    charts.algorithm = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [
                { label: 'Mean Reward', data: rows.map(r => Number(r.mean_reward || 0)), backgroundColor: 'rgba(16, 185, 129, 0.7)' },
                { label: 'Mean Cost', data: rows.map(r => Number(r.mean_cost || 0)), backgroundColor: 'rgba(245, 158, 11, 0.7)' },
                { label: 'Mean Peak Demand', data: rows.map(r => Number(r.mean_peak_demand || 0)), backgroundColor: 'rgba(37, 99, 235, 0.7)' },
                { label: 'Mean Comfort', data: rows.map(r => Number(r.mean_comfort || 0)), backgroundColor: 'rgba(168, 85, 247, 0.7)' }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { labels: { color: '#f1f5f9' } } },
            scales: {
                y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(148, 163, 184, 0.1)' } },
                x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(148, 163, 184, 0.1)' } }
            }
        }
    });
}

async function loadAblationSummary() {
    try {
        const response = await fetch('/api/ablation_summary');
        const data = await response.json();
        if (!data.available) {
            addLogEntry(`Ablation results unavailable. ${data.hint || ''}`);
            showChartMessage('ablationChart', `Ablation study data is not available yet. ${data.hint || 'Run the ablation study to populate this chart.'}`);
            return;
        }
        const rows = data.rows || [];
        if (!rows.length) {
            showChartMessage('ablationChart', 'Ablation results file is present but contains no rows.');
            return;
        }
        createAblationChart(rows);
    } catch (error) {
        console.error('Error loading ablation summary:', error);
    }
}

function createAblationChart(rows) {
    const ctx = document.getElementById('ablationChart');
    if (!ctx) return;
    clearChartMessage('ablationChart');
    if (charts.ablation) charts.ablation.destroy();

    charts.ablation = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: rows.map(r => r.setting),
            datasets: [
                { label: 'Peak Demand (kW)', data: rows.map(r => Number(r.mean_peak_demand_kW || 0)), backgroundColor: 'rgba(37, 99, 235, 0.7)' },
                { label: 'Total Cost (USD)', data: rows.map(r => Number(r.mean_total_cost_usd || 0)), backgroundColor: 'rgba(245, 158, 11, 0.7)' },
                { label: 'Min Comfort', data: rows.map(r => Number(r.mean_min_comfort || 0)), backgroundColor: 'rgba(168, 85, 247, 0.7)' }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { labels: { color: '#f1f5f9' } } },
            scales: {
                y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(148, 163, 184, 0.1)' } },
                x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(148, 163, 184, 0.1)' } }
            }
        }
    });
}

async function loadParetoData() {
    try {
        const response = await fetch('/api/pareto_data');
        const data = await response.json();
        if (!data.available) {
            addLogEntry(`Pareto data unavailable. ${data.hint || ''}`);
            showChartMessage('paretoChart', `Pareto data is not available yet. ${data.hint || 'Run the Pareto sweep to populate this chart.'}`);
            return;
        }
        if (!Array.isArray(data.results) || data.results.length === 0) {
            showChartMessage('paretoChart', 'Pareto data file is present but contains no results.');
            return;
        }
        createParetoChart(data.results);
    } catch (error) {
        console.error('Error loading pareto data:', error);
    }
}

function createParetoChart(results) {
    const ctx = document.getElementById('paretoChart');
    if (!ctx) return;
    clearChartMessage('paretoChart');
    if (charts.pareto) charts.pareto.destroy();

    // 2D projection: Cost vs Peak (point color by comfort)
    const sorted = [...results].sort((a, b) => Number(a.metrics.mean_total_cost) - Number(b.metrics.mean_total_cost));
    charts.pareto = new Chart(ctx, {
        type: 'line',
        data: {
            labels: sorted.map(r => `w${r.setting_id}`),
            datasets: [
                {
                    label: 'Mean Cost',
                    data: sorted.map(r => Number(r.metrics.mean_total_cost || 0)),
                    borderColor: 'rgb(245, 158, 11)',
                    backgroundColor: 'rgba(245, 158, 11, 0.2)',
                    yAxisID: 'y',
                    tension: 0.25
                },
                {
                    label: 'Mean Peak Demand',
                    data: sorted.map(r => Number(r.metrics.mean_peak_demand || 0)),
                    borderColor: 'rgb(37, 99, 235)',
                    backgroundColor: 'rgba(37, 99, 235, 0.2)',
                    yAxisID: 'y1',
                    tension: 0.25
                },
                {
                    label: 'Mean Min Comfort',
                    data: sorted.map(r => Number(r.metrics.mean_min_comfort || 0)),
                    borderColor: 'rgb(168, 85, 247)',
                    backgroundColor: 'rgba(168, 85, 247, 0.2)',
                    yAxisID: 'y2',
                    tension: 0.25
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { labels: { color: '#f1f5f9' } } },
            scales: {
                x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(148, 163, 184, 0.1)' } },
                y: { position: 'left', ticks: { color: '#94a3b8' }, grid: { color: 'rgba(148, 163, 184, 0.1)' } },
                y1: { position: 'right', ticks: { color: '#94a3b8' }, grid: { drawOnChartArea: false } },
                y2: { display: false }
            }
        }
    });
}

async function loadFinalSummary() {
    try {
        const response = await fetch('/api/final_summary');
        const data = await response.json();
        const container = document.getElementById('finalSummaryTable');
        if (!container) return;
        if (!data.available) {
            container.innerHTML = `<div style="color: var(--warning)">Final summary not available. ${data.hint || ''}</div>`;
            addLogEntry(`Final summary unavailable. ${data.hint || ''}`);
            return;
        }
        const rows = data.rows || [];
        if (!rows.length) {
            container.innerHTML = '<div style="color: var(--warning)">Final summary is empty.</div>';
            return;
        }

        const columns = Object.keys(rows[0]);
        let html = '<div style="overflow-x:auto;"><table style="width:100%; border-collapse: collapse; font-size: 0.85rem;">';
        html += '<thead><tr>' + columns.map(c => `<th style="text-align:left; padding:8px; border-bottom:1px solid var(--border);">${c}</th>`).join('') + '</tr></thead><tbody>';
        rows.forEach(row => {
            html += '<tr>' + columns.map(c => `<td style="padding:8px; border-bottom:1px solid rgba(148,163,184,0.15)">${row[c] ?? ''}</td>`).join('') + '</tr>';
        });
        html += '</tbody></table></div>';
        container.innerHTML = html;
    } catch (error) {
        console.error('Error loading final summary:', error);
    }
}

// Action log functions
function addLogEntry(message, type = 'info') {
    const logContainer = document.getElementById('actionLog');
    if (!logContainer) return;
    
    const now = new Date();
    const timeStr = now.toLocaleTimeString('en-US', { hour12: false });
    
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    
    const color = type === 'error' ? 'var(--danger)' : 'var(--text-secondary)';
    
    entry.innerHTML = `
        <span class="log-time" style="color: ${color};">${timeStr}</span>
        <span class="log-message">${message}</span>
    `;
    
    logContainer.insertBefore(entry, logContainer.firstChild);
    
    // Keep only last 50 entries
    while (logContainer.children.length > 50) {
        logContainer.removeChild(logContainer.lastChild);
    }
}

// Export data handler
function handleExportData() {
    addLogEntry('Export functionality: Download results/logs/summary.csv for detailed metrics.');
    // In a real system, this would trigger a download
    alert('Export feature: Results are available in results/logs/summary.csv');
}

// Image modal functions
function openImageModal(src, alt) {
    const modal = document.getElementById('imageModal');
    const modalImg = document.getElementById('modalImage');
    const caption = document.getElementById('modalCaption');
    
    if (modal && modalImg) {
        modalImg.src = src;
        if (caption) caption.textContent = alt || 'Visualization';
        modal.classList.add('show');
        document.body.style.overflow = 'hidden';
    }
}

function closeImageModal() {
    const modal = document.getElementById('imageModal');
    if (modal) {
        modal.classList.remove('show');
        document.body.style.overflow = 'auto';
    }
}

// Make functions globally available
window.openImageModal = openImageModal;
window.closeImageModal = closeImageModal;
