/**
 * Main JavaScript for AI Resume Corrector
 */

document.addEventListener('DOMContentLoaded', function() {
    // File input validation
    const fileInputs = document.querySelectorAll('input[type="file"]');
    
    fileInputs.forEach(input => {
        input.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                // Check file size (5MB limit)
                const fileSize = file.size / 1024 / 1024; // Convert to MB
                if (fileSize > 5) {
                    alert('File size exceeds 5MB limit. Please choose a smaller file.');
                    e.target.value = ''; // Clear the file input
                    return;
                }
                
                // Check file extension
                const fileName = file.name;
                const fileExt = fileName.split('.').pop().toLowerCase();
                
                if (!['pdf', 'docx', 'txt'].includes(fileExt)) {
                    alert('Invalid file type. Please upload a PDF, DOCX, or TXT file.');
                    e.target.value = ''; // Clear the file input
                }
            }
        });
    });
    
    // Form submission loading state
    const forms = document.querySelectorAll('form');
    
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            // Check if this is a form that should show loading state
            // (exclude download form, for example)
            if (!form.id || form.id !== 'downloadForm') {
                const submitBtn = form.querySelector('button[type="submit"]');
                
                if (submitBtn && form.checkValidity()) {
                    // Add loading state
                    const originalText = submitBtn.innerHTML;
                    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
                    submitBtn.disabled = true;
                    
                    // Add loading class to the button
                    submitBtn.classList.add('disabled');
                    
                    // Revert back after timeout (in case of errors)
                    setTimeout(() => {
                        if (submitBtn.disabled) {
                            submitBtn.innerHTML = originalText;
                            submitBtn.disabled = false;
                            submitBtn.classList.remove('disabled');
                        }
                    }, 30000); // 30 seconds timeout
                }
            }
        });
    });
    
    // Enable Bootstrap tooltips
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    if (tooltipTriggerList.length > 0) {
        [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl));
    }
});
