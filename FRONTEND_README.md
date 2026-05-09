# Frontend Web Interface

## Overview

A modern, interactive web interface for presenting the MO-RL-PeakShaving project. The frontend displays:

- Project overview and features
- Performance results and metrics
- Interactive charts and visualizations
- Comparison between RL agent and baseline
- Training progress visualization
- 24-hour episode analysis

## Features

### 🎨 Modern UI
- Responsive design (works on desktop, tablet, mobile)
- Professional gradient header
- Smooth scrolling navigation
- Card-based layout with hover effects

### 📊 Interactive Charts
- **Training Progress**: Episode rewards over time
- **Comparison Chart**: RL vs Baseline metrics
- **Episode Analysis**: 24-hour detailed metrics

### 📈 Visualizations
- Grid Import vs PV Generation
- Battery State of Charge
- Consumer Comfort Score
- Overview Dashboard

### 📱 Responsive Design
- Mobile-friendly layout
- Adaptive grid system
- Touch-friendly navigation

## Installation

1. Install Flask dependencies:
```bash
pip install flask flask-cors
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

## Running the Frontend

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. The interface will automatically load:
   - Project information
   - Evaluation results
   - Training statistics
   - Episode traces
   - Visualization plots

## Project Structure

```
MO-RL-PeakShaving/
├── app.py                 # Flask backend server
├── templates/
│   └── index.html        # Main HTML page
├── static/
│   ├── style.css         # CSS styling
│   └── app.js            # JavaScript for interactivity
└── results/              # Data and plots (generated)
    ├── logs/
    └── plots/
```

## API Endpoints

- `GET /` - Main page
- `GET /api/summary` - Evaluation summary (CSV data)
- `GET /api/episode_trace` - RL episode trace data
- `GET /api/training_stats` - Training statistics
- `GET /api/project_info` - Project information
- `GET /api/plots/<name>` - Serve plot images
- `GET /api/models/status` - Check model availability

## Technologies Used

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **Charts**: Chart.js
- **Styling**: Custom CSS with modern design

## Features in Detail

### 1. Overview Section
- Problem statement
- Solution approach
- Key features list
- Available actions

### 2. Results Section
- Side-by-side comparison cards
- Color-coded metrics (green = better, red = worse)
- Real-time data from evaluation

### 3. Metrics Section
- Explanation of why no accuracy metric
- RL vs traditional ML comparison
- Metrics explanation

### 4. Training Progress
- Line chart showing learning curve
- Smoothed reward trend
- Episode-by-episode progress

### 5. Comparison Chart
- Bar chart comparing all metrics
- RL Agent vs Baseline
- Interactive tooltips

### 6. Visualizations
- Static plot images
- Grid import analysis
- Battery SOC tracking
- Comfort score monitoring

### 7. Episode Analysis
- Multi-axis line chart
- PV, Grid Import, SOC, Comfort
- 24-hour timeline

## Customization

### Change Colors
Edit `static/style.css`:
```css
:root {
    --primary-color: #2563eb;  /* Change this */
    --secondary-color: #1e40af;
    /* ... */
}
```

### Add New Charts
Edit `static/app.js` and add new Chart.js instances.

### Modify Layout
Edit `templates/index.html` to rearrange sections.

## Troubleshooting

### No data showing?
- Ensure evaluation has been run: `python evaluation/evaluate.py`
- Check that `results/logs/summary.csv` exists
- Verify plots exist in `results/plots/`

### Charts not loading?
- Check browser console for errors
- Ensure Chart.js CDN is accessible
- Verify data format matches expected structure

### Images not displaying?
- Check that plots were generated
- Verify file paths in `results/plots/`
- Check browser network tab for 404 errors

## Presentation Tips

1. **Before presenting**:
   - Run full training: `python train/train_ppo.py`
   - Run evaluation: `python evaluation/evaluate.py`
   - Start web server: `python app.py`

2. **During presentation**:
   - Use full-screen browser mode (F11)
   - Navigate sections using top menu
   - Highlight key metrics and comparisons
   - Show interactive charts

3. **Key points to emphasize**:
   - No dataset needed (environment generates data)
   - Performance metrics (not accuracy)
   - RL outperforms baseline
   - Multi-objective optimization

## License

Part of the MO-RL-PeakShaving project.

