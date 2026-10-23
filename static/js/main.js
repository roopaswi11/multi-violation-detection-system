// Traffic Guard AI - Frontend Interactivity & Detection Pipeline

document.addEventListener('DOMContentLoaded', function () {
    // 1. Initialize Dashboard Charts if canvas elements exist
    initDashboardCharts();

    // 2. Setup Upload & Detection Interface
    setupUploadPage();
});

// ---------------- Dashboard Visualization ----------------
function initDashboardCharts() {
    const pieCanvas = document.getElementById('violationPieChart');
    if (pieCanvas) {
        fetch('/api/stats')
            .then(res => res.json())
            .then(stats => {
                const noHelmet = stats.no_helmet_count || 0;
                const triple = stats.triple_riding_count || 0;
                const wrongSide = stats.wrong_side_count || 0;
                const total = stats.total_violations || 0;

                new Chart(pieCanvas, {
                    type: 'doughnut',
                    data: {
                        labels: ['No Helmet', 'Triple Riding', 'Wrong-Side Riding'],
                        datasets: [{
                            data: total > 0 ? [noHelmet, triple, wrongSide] : [1, 1, 1],
                            backgroundColor: ['#ef4444', '#f59e0b', '#dc2626'],
                            borderWidth: 0,
                            hoverOffset: 6
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: {
                                position: 'bottom',
                                labels: { color: '#94a3b8', font: { family: 'Inter', size: 12 } }
                            }
                        },
                        cutout: '72%'
                    }
                });
            })
            .catch(err => console.error("Error loading stats for charts:", err));
    }
}

// ---------------- Upload & Detection Interface ----------------
function setupUploadPage() {
    const dropzone = document.getElementById('uploadDropzone');
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browseFilesBtn');
    const previewContainer = document.getElementById('filePreviewContainer');
    const previewName = document.getElementById('previewFileName');
    const previewSize = document.getElementById('previewFileSize');
    const previewIcon = document.getElementById('previewFileIcon');
    const imgPreview = document.getElementById('imagePreview');
    const videoPreview = document.getElementById('videoPreview');
    const clearFileBtn = document.getElementById('clearFileBtn');
    const analyzeActionContainer = document.getElementById('analyzeActionContainer');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const errorBanner = document.getElementById('uploadErrorBanner');
    const errorMessage = document.getElementById('uploadErrorMessage');
    const resultsSection = document.getElementById('resultsSection');
    const loadingSubtext = document.getElementById('loadingSubtext');

    if (!dropzone || !fileInput) return;

    let currentObjectURL = null;
    let selectedFile = null;

    // Trigger file chooser
    if (browseBtn) {
        browseBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            fileInput.click();
        });
    }

    dropzone.addEventListener('click', (e) => {
        // Only trigger if browse button wasn't directly clicked
        if (e.target !== browseBtn && !browseBtn.contains(e.target)) {
            fileInput.click();
        }
    });

    // Drag & drop highlight
    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        });
    });

    dropzone.addEventListener('drop', (e) => {
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileSelection(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files && fileInput.files.length > 0) {
            handleFileSelection(fileInput.files[0]);
        }
    });

    // Handle selected file and generate preview
    function handleFileSelection(file) {
        if (!file) return;
        selectedFile = file;

        // Clean previous preview URL
        if (currentObjectURL) {
            URL.revokeObjectURL(currentObjectURL);
            currentObjectURL = null;
        }

        const fileName = file.name;
        const fileSizeMB = (file.size / (1024 * 1024)).toFixed(2);
        const lowerName = fileName.toLowerCase();
        const isVideo = lowerName.endsWith('.mp4') || lowerName.endsWith('.avi') || lowerName.endsWith('.mov') || lowerName.endsWith('.webm');
        const isImage = lowerName.endsWith('.jpg') || lowerName.endsWith('.jpeg') || lowerName.endsWith('.png') || lowerName.endsWith('.webp');

        if (!isImage && !isVideo) {
            showError(`Unsupported file type (${fileName}). Please select an image (JPG, PNG, WEBP) or video (MP4, AVI, MOV).`);
            return;
        }

        hideError();
        if (resultsSection) {
            resultsSection.style.display = 'none';
            resultsSection.innerHTML = '';
        }

        // Set metadata
        previewName.textContent = fileName;
        previewSize.textContent = `${fileSizeMB} MB • ${isVideo ? 'Surveillance Video' : 'Surveillance Image'}`;
        previewIcon.textContent = isVideo ? '🎬' : '📷';

        currentObjectURL = URL.createObjectURL(file);

        if (isImage) {
            videoPreview.pause();
            videoPreview.removeAttribute('src');
            videoPreview.style.display = 'none';
            imgPreview.src = currentObjectURL;
            imgPreview.style.display = 'block';
        } else {
            imgPreview.style.display = 'none';
            imgPreview.removeAttribute('src');
            videoPreview.src = currentObjectURL;
            videoPreview.style.display = 'block';
        }

        previewContainer.style.display = 'block';
        analyzeActionContainer.style.display = 'block';
        dropzone.style.display = 'none';
    }

    // Clear file selection
    if (clearFileBtn) {
        clearFileBtn.addEventListener('click', resetSelection);
    }

    function resetSelection() {
        if (currentObjectURL) {
            URL.revokeObjectURL(currentObjectURL);
            currentObjectURL = null;
        }
        selectedFile = null;
        fileInput.value = '';

        imgPreview.style.display = 'none';
        imgPreview.removeAttribute('src');
        videoPreview.pause();
        videoPreview.style.display = 'none';
        videoPreview.removeAttribute('src');

        previewContainer.style.display = 'none';
        analyzeActionContainer.style.display = 'none';
        loadingIndicator.style.display = 'none';
        hideError();

        if (resultsSection) {
            resultsSection.style.display = 'none';
            resultsSection.innerHTML = '';
        }

        dropzone.style.display = 'block';
        if (analyzeBtn) {
            analyzeBtn.disabled = false;
        }
    }

    // ANALYZE button click action
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', () => {
            if (!selectedFile) {
                showError("Please select an image or video file first.");
                return;
            }

            const fileName = selectedFile.name.toLowerCase();
            const isVideo = fileName.endsWith('.mp4') || fileName.endsWith('.avi') || fileName.endsWith('.mov') || fileName.endsWith('.webm');

            // 1. Disable repeated submissions
            analyzeBtn.disabled = true;
            hideError();
            if (resultsSection) resultsSection.style.display = 'none';

            // 2. Show loading indicator and message
            loadingIndicator.style.display = 'block';
            if (loadingSubtext) {
                loadingSubtext.textContent = isVideo 
                    ? "Running ByteTrack trajectory consensus, YOLO11, and OCR across frames..."
                    : "Running YOLO11 detection, rider verification, and OCR on static frame...";
            }

            // 3. Prepare upload payload
            const directionSelect = document.getElementById('permitted_direction');
            const selectedDirection = directionSelect ? directionSelect.value : 'auto';

            const formData = new FormData();
            formData.append('file', selectedFile);
            formData.append('permitted_direction', selectedDirection);
            formData.append('ajax', '1');

            // 4. Send request to Flask backend
            fetch('/upload', {
                method: 'POST',
                body: formData,
                headers: {
                    'X-Requested-With': 'XMLHttpRequest'
                }
            })
            .then(async (response) => {
                if (!response.ok) {
                    let errMsg = `Server returned error (${response.status})`;
                    try {
                        const errData = await response.json();
                        if (errData && errData.error) errMsg = errData.error;
                    } catch (e) {}
                    throw new Error(errMsg);
                }
                return response.json();
            })
            .then((data) => {
                loadingIndicator.style.display = 'none';
                analyzeBtn.disabled = false;

                if (data.success) {
                    renderResults(data);
                } else {
                    showError(data.error || "Analysis failed to complete.");
                }
            })
            .catch((err) => {
                console.error("Analysis error:", err);
                loadingIndicator.style.display = 'none';
                analyzeBtn.disabled = false;
                showError(err.message || "Failed to analyze traffic footage. Please try again.");
            });
        });
    }

    function showError(msg) {
        if (errorMessage) errorMessage.textContent = msg;
        if (errorBanner) errorBanner.style.display = 'flex';
    }

    function hideError() {
        if (errorBanner) errorBanner.style.display = 'none';
    }

    // Render Clean Results Section Beneath Analyze Button
    function renderResults(data) {
        if (!resultsSection) return;

        const isVideo = data.file_type === 'video';
        const violations = data.violations || [];
        const totalViolations = violations.length;
        const confirmedCount = violations.filter(v => v.review_status !== 'insufficient_evidence' && v.status !== 'insufficient_evidence').length;
        const candidateCount = totalViolations - confirmedCount;
        const totalDetections = data.total_detections !== undefined && data.total_detections !== null
            ? data.total_detections 
            : (data.diagnostics ? data.diagnostics.total_detections : null);

        let violationsHtml = '';
        if (totalViolations > 0) {
            violationsHtml = `
                <div style="max-height: 580px; overflow-y: auto; padding-right: 0.4rem;">
                    ${violations.map(v => {
                        const isCand = v.review_status === 'insufficient_evidence' || v.is_candidate || v.status === 'insufficient_evidence';
                        const vtype = isCand ? 'WRONG-SIDE RIDING — NEEDS REVIEW' : (v.violation_type || 'violation').toUpperCase().replace('_', ' ');
                        const confDisplay = (v.confidence !== null && v.confidence !== undefined && v.confidence > 0) 
                            ? `${Math.round(v.confidence * 100)}%` 
                            : (v.detector_confidence !== null && v.detector_confidence !== undefined 
                                ? `<span style="color: var(--amber); font-weight: 600;" title="Object Detector Confidence">Det: ${Math.round(v.detector_confidence * 100)}%</span>`
                                : (isCand ? '<span style="color: var(--amber); font-weight: 600;">Uncertain</span>' : 'N/A'));
                        
                        let badgeColor = 'badge-red';
                        let cardClass = '';
                        if (isCand) {
                            badgeColor = 'badge-amber';
                            cardClass = 'wrong-side needs-review';
                        } else if (v.violation_type === 'triple_riding') {
                            badgeColor = 'badge-amber';
                            cardClass = 'triple';
                        } else if (v.violation_type === 'wrong_side') {
                            badgeColor = 'badge-red';
                            cardClass = 'wrong-side';
                        }

                        let statusBadge = '<span class="badge badge-blue">Pending Review</span>';
                        if (isCand) {
                            statusBadge = '<span class="badge badge-amber">INSUFFICIENT EVIDENCE</span>';
                        } else if (v.review_status === 'confirmed') {
                            statusBadge = '<span class="badge badge-green">Confirmed</span>';
                        } else if (v.review_status === 'disputed') {
                            statusBadge = '<span class="badge badge-red">Disputed</span>';
                        }

                        const detailsUrl = v.id ? `/violation/${v.id}` : `/results/${data.job_id}`;

                        return `
                            <div class="violation-card-item ${cardClass}">
                                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.6rem;">
                                    <div>
                                        <span class="badge ${badgeColor}">${vtype}</span>
                                        ${v.track_id ? `<span class="badge badge-blue" style="margin-left: 0.3rem;">Track ID: ${v.track_id}</span>` : ''}
                                    </div>
                                    <div style="text-align: right;">
                                        <div style="font-size: 0.76rem; color: var(--text-muted);">Confidence</div>
                                        <div style="font-weight: 700; color: #fff; font-size: 0.95rem;">${confDisplay}</div>
                                    </div>
                                </div>

                                <div style="display: grid; grid-template-columns: 1fr auto; gap: 0.8rem; align-items: center;">
                                    <div>
                                        ${v.rider_count ? `
                                            <div style="font-size: 0.85rem; color: var(--text-main); margin-bottom: 0.2rem;">
                                                👥 <strong>Riders Detected:</strong> ${v.rider_count} riders on vehicle
                                            </div>
                                        ` : ''}

                                        ${(v.permitted_direction_label || v.permitted_direction) ? `
                                            <div style="font-size: 0.85rem; color: var(--text-main); margin-bottom: 0.2rem;">
                                                🛣️ <strong>Permitted Direction:</strong> ${v.permitted_direction_label || v.permitted_direction}
                                            </div>
                                        ` : ''}

                                        ${v.estimated_orientation ? `
                                            <div style="font-size: 0.85rem; color: var(--text-main); margin-bottom: 0.2rem;">
                                                🧭 <strong>Orientation:</strong> ${v.estimated_orientation}
                                            </div>
                                        ` : (v.estimated_movement ? `
                                            <div style="font-size: 0.85rem; color: var(--text-main); margin-bottom: 0.2rem;">
                                                🧭 <strong>Movement:</strong> ${v.estimated_movement}
                                            </div>
                                        ` : '')}

                                        ${v.plate_text ? `
                                            <div style="font-size: 0.85rem; color: var(--text-main); margin-bottom: 0.2rem;">
                                                🪪 <strong>Plate OCR:</strong>
                                                <span style="background: rgba(255,255,255,0.08); padding: 0.15rem 0.4rem; border-radius: 4px; font-family: monospace; font-weight: 700; color: #e2e8f0;">
                                                    ${v.plate_text}
                                                </span>
                                                ${v.plate_confidence ? `<span style="font-size: 0.75rem; color: var(--text-dim);">(${Math.round(v.plate_confidence * 100)}%)</span>` : ''}
                                            </div>
                                        ` : ''}

                                        ${v.frame_number !== undefined && v.frame_number !== null ? `
                                            <div style="font-size: 0.78rem; color: var(--text-dim);">
                                                Frame: ${v.frame_number} &bull; Timestamp: ${v.timestamp_sec || 0}s
                                            </div>
                                        ` : ''}

                                        ${(v.reason || v.reviewer_notes) ? `
                                            <div style="font-size: 0.78rem; color: var(--text-muted); margin-top: 0.3rem; font-style: italic;">
                                                Note: ${v.reviewer_notes || v.reason}
                                            </div>
                                        ` : ''}
                                    </div>

                                    ${v.evidence_path ? `
                                        <div>
                                            <a href="${detailsUrl}">
                                                <img src="${v.evidence_path}" class="evidence-thumb-img" alt="Violation Crop">
                                            </a>
                                        </div>
                                    ` : ''}
                                </div>

                                <div style="margin-top: 0.8rem; padding-top: 0.6rem; border-top: 1px solid rgba(255,255,255,0.05); display: flex; justify-content: space-between; align-items: center;">
                                    ${statusBadge}
                                    <a href="${detailsUrl}" class="btn btn-secondary" style="padding: 0.25rem 0.7rem; font-size: 0.78rem;">
                                        Review & Details &rarr;
                                    </a>
                                </div>
                            </div>
                        `;
                    }).join('')}
                </div>
            `;
        } else {
            // Compliant or No objects
            if (totalDetections === 0) {
                violationsHtml = `
                    <div style="text-align: center; padding: 2.5rem 1rem; background: rgba(245, 158, 11, 0.05); border: 1px dashed rgba(245, 158, 11, 0.3); border-radius: 8px;">
                        <div style="font-size: 2.2rem; color: #fbbf24; margin-bottom: 0.5rem;">⚠️</div>
                        <div style="font-size: 1.1rem; color: #fff; font-weight: 600;">No Traffic Objects Detected</div>
                        <p style="font-size: 0.85rem; color: var(--text-dim); margin-top: 0.4rem; line-height: 1.4;">
                            The detector did not locate any motorcycles, riders, or helmets in this footage.
                        </p>
                    </div>
                `;
            } else {
                violationsHtml = `
                    <div style="text-align: center; padding: 2.5rem 1rem; color: var(--text-dim);">
                        <div style="font-size: 2.2rem; color: #34d399; margin-bottom: 0.5rem;">✓</div>
                        <div style="font-size: 1.1rem; color: #fff; font-weight: 600;">No Traffic Violations Detected</div>
                        <p style="font-size: 0.85rem; margin-top: 0.4rem;">
                            All observed riders comply with helmet rules and passenger limits.
                        </p>
                        ${totalDetections ? `
                            <div style="margin-top: 0.8rem; font-size: 0.8rem; color: var(--text-muted);">
                                Observed Objects: <strong>${totalDetections}</strong> verified compliant
                            </div>
                        ` : ''}
                    </div>
                `;
            }
        }

        const mediaViewerHtml = isVideo ? `
            <video controls autoplay loop muted style="width: 100%; height: auto; max-height: 520px; border-radius: var(--radius-sm);">
                <source src="${data.output_path}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        ` : `
            <img src="${data.output_path}" alt="Annotated Result" style="width: 100%; height: auto; max-height: 520px; object-fit: contain; border-radius: var(--radius-sm);">
        `;

        let summaryBadgeHtml = '';
        if (confirmedCount > 0 && candidateCount > 0) {
            summaryBadgeHtml = `<span class="badge badge-red">${confirmedCount} Confirmed</span><span class="badge badge-amber">${candidateCount} Needs Review</span>`;
        } else if (confirmedCount > 0) {
            summaryBadgeHtml = `<span class="badge badge-red">${confirmedCount} Violations</span>`;
        } else if (candidateCount > 0) {
            summaryBadgeHtml = `<span class="badge badge-amber">${candidateCount} Needs Review</span>`;
        } else {
            summaryBadgeHtml = `<span class="badge badge-green">0 Violations</span>`;
        }

        const fineSummary = data.fine_summary || {};
        const isIncomplete = fineSummary.is_incomplete || false;
        const totalFineAmount = fineSummary.total_fine_amount;
        const confirmedViolationsCount = fineSummary.confirmed_violations_count !== undefined 
            ? fineSummary.confirmed_violations_count 
            : confirmedCount;
        const primaryPlate = fineSummary.primary_plate;
        const breakdown = fineSummary.breakdown || [];

        let totalFineDisplay = 'TOTAL FINE: ₹0';
        let totalFineColor = '#34d399';
        let totalFineBadge = '<span class="badge badge-green">0 Violations Confirmed</span>';
        let totalFineSubtext = 'All detected riders and vehicles comply with traffic safety regulations. Zero statutory penalties assessed.';
        let cardBg = 'linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(30, 41, 59, 0.8) 100%)';
        let cardBorder = 'rgba(16, 185, 129, 0.3)';
        let cardShadow = 'rgba(16, 185, 129, 0.1)';

        if (isIncomplete) {
            totalFineDisplay = 'TOTAL FINE: INCOMPLETE';
            totalFineColor = '#fbbf24';
            totalFineBadge = '<span class="badge badge-amber">Fine Amount Unavailable</span>';
            totalFineSubtext = '<span style="color: #fbbf24; font-weight: 600;">⚠️ Incomplete Total:</span> One or more confirmed violations require fine rule configuration or verification. Missing fine amounts are not silently counted as zero.';
            cardBg = 'linear-gradient(135deg, rgba(245, 158, 11, 0.12) 0%, rgba(30, 41, 59, 0.8) 100%)';
            cardBorder = 'rgba(245, 158, 11, 0.4)';
            cardShadow = 'rgba(245, 158, 11, 0.15)';
        } else if (confirmedViolationsCount > 0 && totalFineAmount !== null && totalFineAmount !== undefined) {
            totalFineDisplay = `TOTAL FINE: ₹${Math.round(totalFineAmount).toLocaleString()}`;
            totalFineColor = '#34d399';
            totalFineBadge = `<span class="badge badge-green">${confirmedViolationsCount} Confirmed Violation${confirmedViolationsCount !== 1 ? 's' : ''}</span>`;
            totalFineSubtext = `Calculated strictly from ${confirmedViolationsCount} distinct verified confirmed violation${confirmedViolationsCount !== 1 ? 's' : ''}.`;
            cardBg = 'linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 182, 212, 0.1) 50%, rgba(15, 23, 42, 0.9) 100%)';
            cardBorder = 'rgba(16, 185, 129, 0.45)';
            cardShadow = 'rgba(16, 185, 129, 0.25)';
        }

        let breakdownHtml = '';
        if (breakdown.length > 0) {
            breakdownHtml = `
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 0.8rem; margin-top: 1rem;">
                    ${breakdown.map(b => {
                        const subtotalDisplay = b.has_unavailable 
                            ? '<span style="color: #fbbf24; font-weight: 700; font-size: 0.9rem;">Fine amount unavailable</span>'
                            : `<span style="color: #34d399; font-weight: 800; font-size: 1.2rem;">₹${Math.round(b.subtotal).toLocaleString()}</span>`;
                        const bBadge = b.violation_type === 'triple_riding' ? 'badge-amber' : 'badge-red';
                        return `
                            <div style="background: rgba(0, 0, 0, 0.25); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 6px; padding: 0.8rem 1rem;">
                                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.3rem;">
                                    <div>
                                        <span class="badge ${bBadge}">${(b.short_name || b.violation_type).toUpperCase()}</span>
                                        <div style="font-size: 0.9rem; font-weight: 700; color: #fff; margin-top: 0.3rem;">${b.display_name}</div>
                                    </div>
                                    <span class="badge badge-blue">${b.count} Violation${b.count !== 1 ? 's' : ''}</span>
                                </div>
                                <div style="font-size: 0.74rem; color: var(--text-dim); margin-bottom: 0.4rem;">⚖️ ${b.legal_provision || 'Motor Vehicles Act'}</div>
                                <div style="display: flex; justify-content: space-between; align-items: baseline; border-top: 1px solid rgba(255, 255, 255, 0.06); padding-top: 0.4rem;">
                                    <span style="font-size: 0.75rem; color: var(--text-muted);">${b.unit_amount ? `${b.count} × ₹${Math.round(b.unit_amount).toLocaleString()}` : 'Statutory Rate'}</span>
                                    ${subtotalDisplay}
                                </div>
                            </div>
                        `;
                    }).join('')}
                </div>
            `;
        } else {
            breakdownHtml = `
                <div style="background: rgba(16, 185, 129, 0.05); border: 1px dashed rgba(16, 185, 129, 0.3); border-radius: 6px; padding: 0.8rem; text-align: center; color: var(--text-dim); font-size: 0.82rem; margin-top: 0.8rem;">
                    🛡️ No confirmed violations detected in this analyzed footage. Category penalties: <strong>₹0</strong>.
                </div>
            `;
        }

        const totalFineBannerHtml = `
            <div style="background: ${cardBg}; border: 1px solid ${cardBorder}; box-shadow: 0 10px 30px -10px ${cardShadow}; border-radius: var(--radius-sm); padding: 1.2rem 1.5rem; margin-bottom: 1.5rem;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1.2rem; padding-bottom: 1.2rem; border-bottom: 1px solid rgba(255, 255, 255, 0.08);">
                    <div>
                        <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.3rem;">
                            <span style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 700; color: ${totalFineColor};">
                                Official Statutory Fine Assessment &bull; Motor Vehicles (Amendment) Act 2019
                            </span>
                            ${totalFineBadge}
                        </div>
                        <div style="font-size: 2.3rem; font-family: var(--font-heading); font-weight: 900; line-height: 1.1; color: ${totalFineColor};">
                            ${totalFineDisplay}
                        </div>
                        <p style="font-size: 0.84rem; color: var(--text-dim); margin-top: 0.4rem; margin-bottom: 0;">
                            ${totalFineSubtext}
                        </p>
                    </div>

                    <div style="display: flex; gap: 0.8rem; flex-wrap: wrap;">
                        <div style="background: rgba(0, 0, 0, 0.35); border: 1px solid rgba(255, 255, 255, 0.08); padding: 0.6rem 1rem; border-radius: 6px; text-align: center; min-width: 110px;">
                            <div style="font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; font-weight: 600;">Confirmed</div>
                            <div style="font-size: 1.5rem; font-weight: 800; color: ${confirmedViolationsCount > 0 ? '#f87171' : '#34d399'}; line-height: 1.2; margin-top: 0.1rem;">
                                ${confirmedViolationsCount}
                            </div>
                        </div>

                        <div style="background: rgba(0, 0, 0, 0.35); border: 1px solid rgba(255, 255, 255, 0.08); padding: 0.6rem 1rem; border-radius: 6px; text-align: center; min-width: 140px;">
                            <div style="font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; font-weight: 600;">Vehicle Registration</div>
                            <div style="margin-top: 0.2rem;">
                                ${primaryPlate && primaryPlate !== 'UNREADABLE' 
                                    ? `<span style="background: #fff; color: #000; font-family: monospace; font-weight: 800; font-size: 0.95rem; padding: 0.15rem 0.5rem; border-radius: 3px; border: 1px solid #000; letter-spacing: 0.05em;">🇮🇳 ${primaryPlate}</span>` 
                                    : `<span class="badge badge-amber" style="font-family: monospace; font-size: 0.78rem;">UNREADABLE</span>`}
                            </div>
                        </div>

                        <div style="background: rgba(0, 0, 0, 0.35); border: 1px solid rgba(255, 255, 255, 0.08); padding: 0.6rem 1rem; border-radius: 6px; text-align: center; min-width: 110px;">
                            <div style="font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; font-weight: 600;">Fine Records</div>
                            <div style="font-size: 1.5rem; font-weight: 800; color: #38bdf8; line-height: 1.2; margin-top: 0.1rem;">
                                ${data.total_fines || 0}
                            </div>
                        </div>
                    </div>
                </div>

                <div style="margin-top: 1rem;">
                    <div style="font-size: 0.78rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700;">
                        Breakdown by Violation Type
                    </div>
                    ${breakdownHtml}
                </div>
            </div>
        `;

        resultsSection.innerHTML = `
            <div class="panel" style="animation: fadeIn 0.4s ease;">
                <!-- Header with Action Buttons -->
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem; flex-wrap: wrap; gap: 1rem;">
                    <div>
                        <div style="display: flex; align-items: center; gap: 0.8rem;">
                            <h2 style="font-family: var(--font-heading); font-size: 1.5rem; color: #fff; margin: 0;">Analysis Results</h2>
                            <span class="badge ${isVideo ? 'badge-purple' : 'badge-blue'}">${data.file_type.toUpperCase()}</span>
                            ${summaryBadgeHtml}
                        </div>
                        <div style="font-size: 0.82rem; color: var(--text-muted); margin-top: 0.3rem;">
                            File: <strong>${data.original_filename}</strong> &bull; Job ID: <code>#${data.job_id}</code>
                        </div>
                    </div>
                    <div style="display: flex; gap: 0.8rem;">
                        ${data.download_url ? `
                            <a href="${data.download_url}" class="btn btn-primary" style="font-size: 0.88rem; padding: 0.6rem 1.1rem;">
                                ⭳ Download Annotated ${isVideo ? 'Video' : 'Image'}
                            </a>
                        ` : ''}
                        <a href="/results/${data.job_id}" class="btn btn-secondary" style="font-size: 0.88rem; padding: 0.6rem 1.1rem;">
                            Full Report &rarr;
                        </a>
                    </div>
                </div>

                <!-- Prominent Hero Total Fine Assessment -->
                ${totalFineBannerHtml}

                <!-- Grid: Annotated Media & Violations List -->
                <div class="results-grid" style="margin-bottom: 0;">
                    <!-- Left: Media Viewer -->
                    <div>
                        <div style="background: #000; border: 1px solid var(--border-color); border-radius: var(--radius-sm); padding: 0.5rem; display: flex; justify-content: center; align-items: center; min-height: 280px;">
                            ${mediaViewerHtml}
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 0.8rem; font-size: 0.82rem; color: var(--text-dim);">
                            <div>Models: <span style="color: #93c5fd;">best.pt</span> + <span style="color: #a78bfa;">yolov11n.pt</span></div>
                            <div>Status: <span style="color: #34d399; font-weight: 600;">COMPLETED</span></div>
                        </div>
                    </div>

                    <!-- Right: Violations Cards -->
                    <div>
                        <div style="font-family: var(--font-heading); font-size: 1.1rem; font-weight: 600; color: #fff; margin-bottom: 1rem; display: flex; justify-content: space-between;">
                            <span>Detected Violations (${totalViolations})</span>
                            <a href="/history" style="font-size: 0.8rem; color: var(--cyan-accent); text-decoration: none;">History &rarr;</a>
                        </div>
                        ${violationsHtml}
                    </div>
                </div>
            </div>
        `;

        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}
