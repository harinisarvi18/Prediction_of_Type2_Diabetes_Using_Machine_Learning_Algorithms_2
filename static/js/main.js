window.addEventListener('DOMContentLoaded', () => {
    const div = document.getElementById('patientData');
    if (!div) return;

    const patientData = JSON.parse(div.getAttribute('data-values'));

    const { systolic_bp, diastolic_bp, glucose, bmi, probability } = patientData;

    // Animate progress bar
    const progressBar = document.getElementById('riskProgress');
    if (progressBar) {
        let width = 0;
        const interval = setInterval(() => {
            if (width >= probability) clearInterval(interval);
            else { width++; progressBar.style.width = width + '%'; }
        }, 10);
    }

    // Draw chart
    const chartCanvas = document.getElementById('patientChart');
    if (chartCanvas) {
        new Chart(chartCanvas.getContext('2d'), {
            type: 'bar',
            data: {
                labels: ['Systolic BP', 'Diastolic BP', 'Glucose', 'BMI'],
                datasets: [{
                    label: 'Your Values',
                    data: [systolic_bp, diastolic_bp, glucose, bmi],
                    backgroundColor: ['#0d6efd', '#198754', '#ffc107', '#dc3545']
                }]
            },
            options: { responsive: true, plugins: { legend: { display: false } } }
        });
    }
});
