document.addEventListener('DOMContentLoaded', function() {
    // Add animation to all cards
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        card.style.animationDelay = `${index * 0.1}s`;
    });
    
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Add active class to current nav item
    const currentPage = location.pathname.split('/').pop() || 'index';
    document.querySelectorAll('.nav-link').forEach(link => {
        if(link.getAttribute('href').endsWith(currentPage)) {
            link.classList.add('active');
            link.setAttribute('aria-current', 'page');
        }
    });
    
    // Add confirmation for file upload replacement
    const fileInput = document.getElementById('file');
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            const fileName = this.files[0]?.name;
            if (fileName) {
                console.log(`File selected: ${fileName}`);
            }
        });
    }
});