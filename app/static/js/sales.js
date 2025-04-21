// sales.js - Sales Forecasting Dashboard JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all charts with error handling
    try {
        if (document.getElementById('salesChart') && document.getElementById('forecastChart')) {
            initSalesChart();
            initForecastChart();
        }
    } catch (error) {
        console.error("Error initializing charts:", error);
        showAlert("Failed to initialize charts. Please try again.", "danger");
    }
    
    // Set up event listeners
    setupExportModal();
    setupCopyButton();
    
    // Set default dates in export modal
    setDefaultExportDates();
});

/**
 * Initialize the Actual vs Predicted Sales Chart
 */
function initSalesChart() {
    const chartDataElement = document.getElementById('chartData');
    if (!chartDataElement) {
        console.error("Chart data element not found");
        return;
    }

    try {
        const graphData = JSON.parse(chartDataElement.textContent);
        const ctx = document.getElementById('salesChart').getContext('2d');
        
        // Format dates if they exist in the labels
        const labels = graphData.actual_vs_predicted.labels.map(label => {
            if (typeof label === 'string' && label.match(/\d{4}-\d{2}-\d{2}/)) {
                return new Date(label).toLocaleDateString();
            }
            return label;
        });
        
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Actual Sales',
                        data: graphData.actual_vs_predicted.actual,
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        borderWidth: 2,
                        tension: 0.1,
                        fill: false,
                        pointRadius: 0,
                        pointHoverRadius: 5
                    },
                    {
                        label: 'Predicted Sales',
                        data: graphData.actual_vs_predicted.predicted,
                        borderColor: '#e74c3c',
                        backgroundColor: 'rgba(231, 76, 60, 0.1)',
                        borderWidth: 2,
                        tension: 0.1,
                        fill: false,
                        pointRadius: 0,
                        pointHoverRadius: 5,
                        borderDash: [5, 5]
                    }
                ]
            },
            options: getChartOptions('Actual vs Predicted Sales', 'Sales Value')
        });
    } catch (error) {
        console.error("Error initializing sales chart:", error);
        throw error;
    }
}

/**
 * Initialize the Forecast Chart with Confidence Intervals
 */
function initForecastChart() {
    const chartDataElement = document.getElementById('chartData');
    if (!chartDataElement) {
        console.error("Chart data element not found");
        return;
    }

    try {
        const graphData = JSON.parse(chartDataElement.textContent);
        const ctx = document.getElementById('forecastChart').getContext('2d');
        
        // Format dates if they exist in the labels
        const labels = graphData.forecast.labels.map(label => {
            if (typeof label === 'string' && label.match(/\d{4}-\d{2}-\d{2}/)) {
                return new Date(label).toLocaleDateString();
            }
            return label;
        });
        
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Historical Sales',
                        data: graphData.forecast.historical,
                        borderColor: '#2ecc71',
                        backgroundColor: 'rgba(46, 204, 113, 0.1)',
                        borderWidth: 2,
                        tension: 0.1,
                        fill: false,
                        pointRadius: 0,
                        pointHoverRadius: 5
                    },
                    {
                        label: 'Forecasted Sales',
                        data: graphData.forecast.forecast,
                        borderColor: '#e67e22',
                        backgroundColor: 'rgba(230, 126, 34, 0.1)',
                        borderWidth: 2,
                        tension: 0.1,
                        fill: false,
                        pointRadius: 0,
                        pointHoverRadius: 5
                    },
                    {
                        label: 'Upper Confidence Bound',
                        data: graphData.forecast.upper_bound,
                        borderColor: 'rgba(230, 126, 34, 0.5)',
                        backgroundColor: 'rgba(230, 126, 34, 0.1)',
                        borderWidth: 1,
                        borderDash: [3, 3],
                        tension: 0.1,
                        fill: '+1',
                        pointRadius: 0,
                        pointHoverRadius: 0
                    },
                    {
                        label: 'Lower Confidence Bound',
                        data: graphData.forecast.lower_bound,
                        borderColor: 'rgba(230, 126, 34, 0.5)',
                        backgroundColor: 'rgba(230, 126, 34, 0.1)',
                        borderWidth: 1,
                        borderDash: [3, 3],
                        tension: 0.1,
                        fill: '-1',
                        pointRadius: 0,
                        pointHoverRadius: 0
                    }
                ]
            },
            options: getChartOptions('Sales Forecast with Confidence Interval', 'Sales Value')
        });
    } catch (error) {
        console.error("Error initializing forecast chart:", error);
        throw error;
    }
}

/**
 * Common chart configuration options
 */
function getChartOptions(title, yAxisLabel) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            title: {
                display: true,
                text: title,
                font: {
                    size: 16,
                    weight: '600'
                },
                padding: {
                    top: 10,
                    bottom: 20
                }
            },
            tooltip: {
                mode: 'index',
                intersect: false,
                backgroundColor: 'rgba(0, 0, 0, 0.8)',
                titleFont: {
                    size: 14,
                    weight: 'bold'
                },
                bodyFont: {
                    size: 12
                },
                padding: 12,
                cornerRadius: 6,
                displayColors: true,
                callbacks: {
                    label: function(context) {
                        let label = context.dataset.label || '';
                        if (label) {
                            label += ': ';
                        }
                        if (context.parsed.y !== null) {
                            label += new Intl.NumberFormat('en-US', {
                                style: 'currency',
                                currency: 'USD'
                            }).format(context.parsed.y);
                        }
                        return label;
                    }
                }
            },
            legend: {
                position: 'top',
                labels: {
                    boxWidth: 12,
                    padding: 20,
                    usePointStyle: true,
                    pointStyle: 'circle'
                }
            }
        },
        scales: {
            x: {
                grid: {
                    display: false
                },
                ticks: {
                    maxRotation: 45,
                    minRotation: 45
                }
            },
            y: {
                beginAtZero: false,
                grid: {
                    color: 'rgba(0, 0, 0, 0.05)'
                },
                title: {
                    display: true,
                    text: yAxisLabel,
                    font: {
                        weight: 'bold'
                    }
                },
                ticks: {
                    callback: function(value) {
                        return new Intl.NumberFormat('en-US', {
                            style: 'currency',
                            currency: 'USD',
                            maximumFractionDigits: 0
                        }).format(value);
                    }
                }
            }
        },
        interaction: {
            intersect: false,
            mode: 'index'
        },
        animation: {
            duration: 1000
        }
    };
}

/**
 * Set up the export modal functionality
 */
function setupExportModal() {
    // Format selection
    document.querySelectorAll('.format-option').forEach(option => {
        option.addEventListener('click', function() {
            document.querySelectorAll('.format-option').forEach(opt => {
                opt.classList.remove('active');
            });
            this.classList.add('active');
            document.getElementById('exportFormat').value = 
                this.querySelector('span').textContent.toLowerCase();
        });
    });
    
    // Initialize with CSV selected
    document.querySelector('.format-option').click();
}

/**
 * Set up the copy summary button
 */
function setupCopyButton() {
    const copyBtn = document.querySelector('.copy-btn');
    if (copyBtn) {
        copyBtn.addEventListener('click', copySummary);
    }
}

/**
 * Copy model summary to clipboard
 */
function copySummary() {
    const summary = document.querySelector('.model-summary pre').textContent;
    navigator.clipboard.writeText(summary).then(() => {
        const originalText = document.querySelector('.copy-btn i').className;
        document.querySelector('.copy-btn i').className = 'fas fa-check';
        document.querySelector('.copy-btn').classList.add('btn-success');
        document.querySelector('.copy-btn').classList.remove('btn-outline-secondary');
        
        setTimeout(() => {
            document.querySelector('.copy-btn i').className = originalText;
            document.querySelector('.copy-btn').classList.remove('btn-success');
            document.querySelector('.copy-btn').classList.add('btn-outline-secondary');
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy text: ', err);
        showAlert("Failed to copy summary to clipboard", "danger");
    });
}

/**
 * Set default dates in export modal
 */
function setDefaultExportDates() {
    const today = new Date();
    const oneYearFromNow = new Date();
    oneYearFromNow.setFullYear(today.getFullYear() + 1);
    
    document.getElementById('startDate').valueAsDate = today;
    document.getElementById('endDate').valueAsDate = oneYearFromNow;
}

/**
 * Export results functionality
 */
function exportResults() {
    const exportModal = new bootstrap.Modal(document.getElementById('exportModal'));
    exportModal.show();
}

/**
 * Confirm export and process
 */
function confirmExport() {
    const format = document.getElementById('exportFormat').value;
    const startDate = document.getElementById('startDate').value;
    const endDate = document.getElementById('endDate').value;
    
    // Show loading state
    const exportBtn = document.querySelector('#exportModal .btn-primary');
    const originalHtml = exportBtn.innerHTML;
    exportBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Exporting...';
    exportBtn.disabled = true;
    
    // Here you would typically make an AJAX call to your backend
    // For demonstration, we'll simulate a delay
    setTimeout(() => {
        // Reset button state
        exportBtn.innerHTML = originalHtml;
        exportBtn.disabled = false;
        
        // Close modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('exportModal'));
        if (modal) {
            modal.hide();
        }
        
        // Show success message
        showAlert(`Successfully exported as ${format.toUpperCase()} format`, 'success');
    }, 1500);
}

/**
 * Show alert message
 */
function showAlert(message, type) {
    // Remove any existing alerts first
    document.querySelectorAll('.alert').forEach(alert => {
        bootstrap.Alert.getInstance(alert)?.close();
    });

    const alert = document.createElement('div');
    alert.className = `alert alert-${type} alert-dismissible fade show`;
    alert.role = 'alert';
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    const container = document.querySelector('.sales-container');
    if (container) {
        container.insertBefore(alert, container.firstChild);
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            const bsAlert = bootstrap.Alert.getInstance(alert);
            if (bsAlert) {
                bsAlert.close();
            }
        }, 5000);
    }
}

/**
 * Format numbers as currency
 */
function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(value);
}