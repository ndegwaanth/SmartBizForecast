document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('uploadForm');
    const fileInput = document.getElementById('fileInput');
    const uploadButton = document.getElementById('uploadButton');
    
    form.addEventListener('submit', function(e) {
        if (!fileInput.files.length) {
            e.preventDefault();
            alert('Please select a file first');
            return false;
        }
        
        uploadButton.disabled = true;
        uploadButton.textContent = 'Uploading...';
        return true;
    });
    
    // Reset button state if form submission fails
    window.addEventListener('pageshow', function() {
        uploadButton.disabled = false;
        uploadButton.textContent = 'Upload';
    });
});