// Password visibility toggle
function togglePassword(inputId, icon) {
    const input = document.getElementById(inputId);
    if (input.type === "password") {
        input.type = "text";
        icon.classList.replace("fa-eye-slash", "fa-eye");
    } else {
        input.type = "password";
        icon.classList.replace("fa-eye", "fa-eye-slash");
    }
}

// Password strength indicator
document.getElementById('floatingPassword')?.addEventListener('input', function(e) {
    const password = e.target.value;
    const strengthBar = document.getElementById('password-strength-bar');
    const strengthText = document.getElementById('password-strength-text');
    
    // Reset
    strengthBar.style.width = '0%';
    strengthBar.className = 'progress-bar';
    
    if (!password) {
        strengthText.textContent = '';
        return;
    }
    
    // Calculate strength (simple version)
    let strength = 0;
    
    // Length
    if (password.length >= 8) strength += 25;
    if (password.length >= 12) strength += 25;
    
    // Complexity
    if (/[A-Z]/.test(password)) strength += 15;
    if (/[0-9]/.test(password)) strength += 15;
    if (/[^A-Za-z0-9]/.test(password)) strength += 20;
    
    // Update UI
    strengthBar.style.width = strength + '%';
    
    if (strength < 40) {
        strengthBar.classList.add('bg-danger');
        strengthText.textContent = 'Weak';
        strengthText.className = 'text-danger';
    } else if (strength < 70) {
        strengthBar.classList.add('bg-warning');
        strengthText.textContent = 'Moderate';
        strengthText.className = 'text-warning';
    } else {
        strengthBar.classList.add('bg-success');
        strengthText.textContent = 'Strong';
        strengthText.className = 'text-success';
    }
});

// Form validation
(function() {
    'use strict';
    
    // Fetch all the forms we want to apply custom Bootstrap validation styles to
    const forms = document.querySelectorAll('.needs-validation');
    
    // Loop over them and prevent submission
    Array.from(forms).forEach(form => {
        form.addEventListener('submit', event => {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            
            form.classList.add('was-validated');
        }, false);
    });
})();
