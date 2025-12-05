/**
 * MedFedAgent Dashboard - Main JavaScript
 * Handles theme switching, real-time updates, and UI interactions
 */

// ============================================================================
// Theme Management
// ============================================================================

const ThemeManager = {
    init() {
        // Check for saved theme preference or default to dark
        const savedTheme = localStorage.getItem('medfedagent-theme') || 'dark';
        this.setTheme(savedTheme);
        
        // Bind theme toggle button
        const toggleBtn = document.getElementById('themeToggle');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', () => this.toggle());
        }
    },
    
    setTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('medfedagent-theme', theme);
        
        // Update toggle button icon
        const toggleBtn = document.getElementById('themeToggle');
        if (toggleBtn) {
            const icon = toggleBtn.querySelector('i');
            if (icon) {
                icon.className = theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
            }
        }
        
        // Update Chart.js defaults for theme
        this.updateChartTheme(theme);
    },
    
    toggle() {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
        this.setTheme(newTheme);
        showToast(`Switched to ${newTheme} mode`, 'info');
    },
    
    updateChartTheme(theme) {
        if (typeof Chart !== 'undefined') {
            const textColor = theme === 'dark' ? 'rgb(156, 163, 175)' : 'rgb(75, 85, 99)';
            const gridColor = theme === 'dark' ? 'rgba(75, 85, 99, 0.3)' : 'rgba(209, 213, 219, 0.5)';
            
            Chart.defaults.color = textColor;
            Chart.defaults.borderColor = gridColor;
        }
    }
};


// ============================================================================
// Toast Notifications
// ============================================================================

function showToast(message, type = 'info', duration = 3000) {
    const container = document.getElementById('toastContainer');
    if (!container) return;
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icons = {
        success: 'fa-check-circle',
        error: 'fa-exclamation-circle',
        info: 'fa-info-circle',
        warning: 'fa-exclamation-triangle'
    };
    
    toast.innerHTML = `
        <i class="fas ${icons[type] || icons.info}"></i>
        <span>${message}</span>
    `;
    
    container.appendChild(toast);
    
    // Remove toast after duration
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease forwards';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// Add slideOut animation
const styleSheet = document.createElement('style');
styleSheet.textContent = `
    @keyframes slideOut {
        from {
            opacity: 1;
            transform: translateX(0);
        }
        to {
            opacity: 0;
            transform: translateX(100%);
        }
    }
`;
document.head.appendChild(styleSheet);


// ============================================================================
// Data Refresh
// ============================================================================

let autoRefreshInterval = null;

async function refreshData() {
    try {
        const response = await fetch('/api/metrics');
        if (!response.ok) throw new Error('Failed to fetch data');
        
        const data = await response.json();
        
        // Update summary metrics
        updateSummaryMetrics(data.summary);
        
        // Update last updated timestamp
        const lastUpdated = document.getElementById('lastUpdated');
        if (lastUpdated) {
            lastUpdated.textContent = data.last_updated;
        }
        
        showToast('Data refreshed', 'success');
        return data;
    } catch (error) {
        console.error('Error refreshing data:', error);
        showToast('Failed to refresh data', 'error');
        return null;
    }
}

function updateSummaryMetrics(summary) {
    if (!summary) return;
    
    // Update individual metric elements
    const elements = {
        'totalRounds': summary.total_rounds,
        'bestAuc': summary.best_auc?.toFixed(4),
        'epsilonSpent': `${(summary.final_epsilon || 0).toFixed(2)} / ${(summary.epsilon_budget || 8).toFixed(1)}`,
        'privacyRemaining': `${(summary.privacy_percentage || 100).toFixed(1)}%`
    };
    
    Object.entries(elements).forEach(([id, value]) => {
        const el = document.getElementById(id);
        if (el && value !== undefined) {
            el.textContent = value;
        }
    });
}

function setupAutoRefresh() {
    const checkbox = document.getElementById('autoRefresh');
    if (!checkbox) return;
    
    checkbox.addEventListener('change', function() {
        if (this.checked) {
            autoRefreshInterval = setInterval(refreshData, 5000);
            showToast('Auto-refresh enabled', 'success');
        } else {
            if (autoRefreshInterval) {
                clearInterval(autoRefreshInterval);
                autoRefreshInterval = null;
            }
            showToast('Auto-refresh disabled', 'info');
        }
    });
}


// ============================================================================
// Chart Utilities
// ============================================================================

const ChartUtils = {
    // Common chart options
    getBaseOptions() {
        return {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 15,
                        font: {
                            size: 12,
                            family: "'Inter', sans-serif"
                        }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    titleFont: { size: 13, weight: '600' },
                    bodyFont: { size: 12 },
                    padding: 12,
                    cornerRadius: 8,
                    displayColors: true
                }
            }
        };
    },
    
    // Create gradient
    createGradient(ctx, color, opacity = 0.4) {
        const gradient = ctx.createLinearGradient(0, 0, 0, 300);
        gradient.addColorStop(0, this.hexToRgba(color, opacity));
        gradient.addColorStop(1, this.hexToRgba(color, 0));
        return gradient;
    },
    
    // Convert hex to rgba
    hexToRgba(hex, alpha) {
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    },
    
    // Format numbers
    formatNumber(value, decimals = 2) {
        if (value === null || value === undefined) return 'N/A';
        return Number(value).toFixed(decimals);
    }
};


// ============================================================================
// Animations & Effects
// ============================================================================

const AnimationUtils = {
    // Counter animation for stat values
    animateCounter(element, target, duration = 1000) {
        const start = parseFloat(element.textContent) || 0;
        const startTime = performance.now();
        
        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // Easing function (ease-out)
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = start + (target - start) * eased;
            
            element.textContent = current.toFixed(target < 1 ? 4 : 0);
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };
        
        requestAnimationFrame(animate);
    },
    
    // Fade in elements on scroll
    initScrollAnimations() {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('fade-in-visible');
                }
            });
        }, {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        });
        
        document.querySelectorAll('.stat-card, .chart-card, .info-card').forEach(el => {
            el.classList.add('fade-in');
            observer.observe(el);
        });
    }
};

// Add fade-in animation styles
const animationStyles = document.createElement('style');
animationStyles.textContent = `
    .fade-in {
        opacity: 0;
        transform: translateY(20px);
        transition: opacity 0.5s ease, transform 0.5s ease;
    }
    
    .fade-in-visible {
        opacity: 1;
        transform: translateY(0);
    }
`;
document.head.appendChild(animationStyles);


// ============================================================================
// Keyboard Shortcuts
// ============================================================================

const KeyboardShortcuts = {
    init() {
        document.addEventListener('keydown', (e) => {
            // Ignore if typing in input
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
            
            // Ctrl/Cmd + R: Refresh data
            if ((e.ctrlKey || e.metaKey) && e.key === 'r') {
                e.preventDefault();
                refreshData();
            }
            
            // T: Toggle theme
            if (e.key === 't' && !e.ctrlKey && !e.metaKey) {
                ThemeManager.toggle();
            }
            
            // 1-5: Navigate to pages
            if (['1', '2', '3', '4', '5'].includes(e.key) && !e.ctrlKey && !e.metaKey) {
                const pages = ['/', '/clinical', '/technical', '/privacy', '/fairness'];
                window.location.href = pages[parseInt(e.key) - 1];
            }
        });
    }
};


// ============================================================================
// Mobile Menu
// ============================================================================

const MobileMenu = {
    init() {
        // Create mobile menu button if doesn't exist
        const navbar = document.querySelector('.navbar');
        if (!navbar) return;
        
        // Check if mobile menu button exists
        let menuBtn = document.querySelector('.mobile-menu-btn');
        if (!menuBtn) {
            menuBtn = document.createElement('button');
            menuBtn.className = 'mobile-menu-btn';
            menuBtn.innerHTML = '<i class="fas fa-bars"></i>';
            menuBtn.style.cssText = `
                display: none;
                background: var(--bg-tertiary);
                border: 1px solid var(--border-color);
                border-radius: var(--radius-lg);
                padding: var(--space-sm);
                color: var(--text-secondary);
                cursor: pointer;
            `;
            
            const navActions = navbar.querySelector('.nav-actions');
            if (navActions) {
                navActions.insertBefore(menuBtn, navActions.firstChild);
            }
        }
        
        menuBtn.addEventListener('click', () => {
            const navLinks = document.querySelector('.nav-links');
            if (navLinks) {
                navLinks.classList.toggle('mobile-visible');
            }
        });
        
        // Add mobile styles
        const mobileStyles = document.createElement('style');
        mobileStyles.textContent = `
            @media (max-width: 768px) {
                .mobile-menu-btn {
                    display: flex !important;
                    align-items: center;
                    justify-content: center;
                }
                
                .nav-links {
                    position: absolute;
                    top: 70px;
                    left: 0;
                    right: 0;
                    background: var(--glass-bg);
                    backdrop-filter: blur(20px);
                    padding: var(--space-md);
                    flex-direction: column;
                    border-bottom: 1px solid var(--glass-border);
                    display: none;
                }
                
                .nav-links.mobile-visible {
                    display: flex;
                }
            }
        `;
        document.head.appendChild(mobileStyles);
    }
};


// ============================================================================
// Utility Functions
// ============================================================================

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(decimals)) + ' ' + sizes[i];
}

function formatDuration(ms) {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    
    if (hours > 0) return `${hours}h ${minutes % 60}m`;
    if (minutes > 0) return `${minutes}m ${seconds % 60}s`;
    return `${seconds}s`;
}


// ============================================================================
// Initialize
// ============================================================================

document.addEventListener('DOMContentLoaded', function() {
    // Initialize theme
    ThemeManager.init();
    
    // Initialize auto-refresh
    setupAutoRefresh();
    
    // Initialize keyboard shortcuts
    KeyboardShortcuts.init();
    
    // Initialize mobile menu
    MobileMenu.init();
    
    // Initialize scroll animations
    AnimationUtils.initScrollAnimations();
    
    console.log('🏥 MedFedAgent Dashboard initialized');
});


// ============================================================================
// Export for use in templates
// ============================================================================

window.MedFedAgent = {
    ThemeManager,
    ChartUtils,
    AnimationUtils,
    showToast,
    refreshData,
    debounce,
    formatBytes,
    formatDuration
};
