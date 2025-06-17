document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('pattern-form');
    const welcomeContainer = document.getElementById('welcome-container');
    const loadingContainer = document.getElementById('loading-container');
    const errorContainer = document.getElementById('error-container');
    const resultsContainer = document.getElementById('results-container');
    const submitButton = document.querySelector('.submit-button');
    const windowRatioSlider = document.getElementById('window-ratio');
    const windowRatioValue = document.getElementById('ratio-value');
    const massingImageInput = document.getElementById('massing-image');
    const generatedImage = document.getElementById('generated-image');
    const metricsContainer = document.getElementById('metrics-container');
    
    let massingImagePreview = null;

    // Initialize slider functionality
    initializeSliders();
    
    // Update slider value displays when they change
    windowRatioSlider.addEventListener('input', (e) => {
        windowRatioValue.textContent = `${e.target.value}%`;
    });

    // Handle file upload preview
    massingImageInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;

        // Remove previous preview if exists
        if (massingImagePreview) {
            massingImagePreview.remove();
            massingImagePreview = null;
        }

        // Create preview
        if (file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = (e) => {
                massingImagePreview = document.createElement('div');
                massingImagePreview.className = 'image-preview';
                massingImagePreview.innerHTML = `
                    <img src="${e.target.result}" alt="Massing preview" style="max-width: 100%; max-height: 200px; margin-top: 1rem;">
                    <button type="button" class="remove-image" style="margin-top: 0.5rem;">Remove Image</button>
                `;
                
                // Add remove button handler
                const removeBtn = massingImagePreview.querySelector('.remove-image');
                removeBtn.addEventListener('click', () => {
                    massingImageInput.value = '';
                    massingImagePreview.remove();
                    massingImagePreview = null;
                });

                massingImageInput.parentNode.appendChild(massingImagePreview);
            };
            reader.readAsDataURL(file);
        }
    });

    // Form submission handler
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Check if massing image is uploaded
        if (!massingImageInput.files || massingImageInput.files.length === 0) {
            showErrorState('Please upload a massing image before generating the facade design.');
            return;
        }
        
        // Disable submit button and change text
        submitButton.disabled = true;
        submitButton.querySelector('.button-text').textContent = 'Generating...';
        
        // Show loading state
        showLoadingState();
        
        try {
            const formData = new FormData(form);
            
            const response = await fetch('/generate', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            // Display successful results
            displayResults(data);
            
        } catch (error) {
            console.error('Error:', error);
            showErrorState(error.message);
        } finally {
            // Re-enable submit button
            submitButton.disabled = false;
            submitButton.querySelector('.button-text').textContent = 'Generate Façade Design';
        }
    });
    
    function showLoadingState() {
        hideAllContainers();
        loadingContainer.style.display = 'flex';
    }
    
    function showErrorState(errorMessage = 'An error occurred while generating the design. Please try again.') {
        hideAllContainers();
        const errorText = document.getElementById('error-text');
        if (errorText) {
            errorText.textContent = errorMessage;
        }
        errorContainer.style.display = 'flex';
    }
    
    function showResultsState() {
        hideAllContainers();
        resultsContainer.style.display = 'block';
    }
    
    function showWelcomeState() {
        hideAllContainers();
        welcomeContainer.style.display = 'flex';
    }
    
    function hideAllContainers() {
        welcomeContainer.style.display = 'none';
        loadingContainer.style.display = 'none';
        errorContainer.style.display = 'none';
        resultsContainer.style.display = 'none';
    }
    
    function displayResults(data) {
        // Display the generated image
        if (data.base64_image) {
            generatedImage.src = `data:image/png;base64,${data.base64_image}`;
            generatedImage.style.display = 'block';
        } else {
            generatedImage.style.display = 'none';
        }
        
        // Display dashboard data
        if (data.dashboard_data) {
            displayDashboard(data.dashboard_data);
        }
        
        // Show results state
        showResultsState();
    }
    
    function displayDashboard(dashboardData) {
        // Create overall performance score (prominent display)
        const overallScoreHtml = `
            <div class="metric-card overall-score">
                <h4 class="metric-label">Overall Performance Score</h4>
                <div class="score-display">
                    <span class="score-value">${dashboardData.overall_performance_score}%</span>
                </div>
                <div class="score-bar">
                    <div class="score-bar-fill ${getScoreClass(dashboardData.overall_performance_score)}" 
                         style="width: ${dashboardData.overall_performance_score}%"></div>
                </div>
            </div>
        `;
        
        // Create individual metrics
        const metricsHtml = `
            <div class="metric-card">
                <h4 class="metric-label">Window-to-Wall Ratio</h4>
                <p class="metric-value">${dashboardData.wwr_actual_percent}%</p>
                <p class="metric-description">Percentage of facade area covered by windows</p>
            </div>
            
            <div class="metric-card">
                <h4 class="metric-label">Daylight Performance</h4>
                <div class="score-display">
                    <span class="metric-value">${dashboardData.daylight_score_estimate}%</span>
                </div>
                <div class="score-bar">
                    <div class="score-bar-fill ${getScoreClass(dashboardData.daylight_score_estimate)}" 
                         style="width: ${dashboardData.daylight_score_estimate}%"></div>
                </div>
                <p class="metric-description">Estimated natural lighting performance</p>
            </div>
            
            <div class="metric-card">
                <h4 class="metric-label">Solar Heat Gain</h4>
                <p class="metric-value">${dashboardData.solar_gain_qualitative}</p>
                <p class="metric-description">Qualitative assessment of solar heat management</p>
            </div>
            
            <div class="metric-card full-width">
                <h4 class="metric-label">Design Description</h4>
                <p class="metric-description">${dashboardData.generated_style_description}</p>
            </div>
            
            <div class="metric-card">
                <h4 class="metric-label">Target WWR</h4>
                <p class="metric-value">${dashboardData.target_wwr}%</p>
                <p class="metric-description">Target window-to-wall ratio</p>
            </div>
            
            <div class="metric-card">
                <h4 class="metric-label">Glass Area</h4>
                <p class="metric-value">${dashboardData.glass_area.toLocaleString()} m²</p>
                <p class="metric-description">Total area of glass in the facade</p>
            </div>
            
            <div class="metric-card">
                <h4 class="metric-label">Wall Area</h4>
                <p class="metric-value">${dashboardData.wall_area.toLocaleString()} m²</p>
                <p class="metric-description">Total area of walls in the facade</p>
            </div>
            
            <div class="metric-card">
                <h4 class="metric-label">Total Facade Area</h4>
                <p class="metric-value">${dashboardData.total_facade_area.toLocaleString()} m²</p>
                <p class="metric-description">Total area of the facade</p>
            </div>
            
            <div class="metric-card">
                <h4 class="metric-label">Building Program</h4>
                <p class="metric-value">${dashboardData.building_program}</p>
                <p class="metric-description">Building program</p>
            </div>
            
            <div class="metric-card">
                <h4 class="metric-label">Climate Type</h4>
                <p class="metric-value">${dashboardData.climate_type}</p>
                <p class="metric-description">Climate type</p>
            </div>
        `;
        
        metricsContainer.innerHTML = overallScoreHtml + metricsHtml;
    }
    
    function getScoreClass(score) {
        if (score >= 80) return 'score-excellent';
        if (score >= 60) return 'score-good';
        if (score >= 40) return 'score-fair';
        return 'score-poor';
    }
    
    function initializeSliders() {
        const sliders = document.querySelectorAll('input[type="range"]');
        
        sliders.forEach(slider => {
            const output = document.getElementById(slider.id + '-output');
            if (output) {
                // Set initial value
                output.textContent = slider.value + '%';
                
                // Update on input
                slider.addEventListener('input', function() {
                    output.textContent = this.value + '%';
                });
            }
        });
    }
    
    // Initialize with welcome state
    showWelcomeState();
});