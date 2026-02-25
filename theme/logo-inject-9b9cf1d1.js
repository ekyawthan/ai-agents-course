// Add logo to sidebar
(function() {
    document.addEventListener('DOMContentLoaded', function() {
        const sidebar = document.querySelector('.sidebar-scrollbox');
        if (!sidebar) return;
        
        // Create logo container
        const logoDiv = document.createElement('div');
        logoDiv.className = 'sidebar-logo';
        
        const logoImg = document.createElement('img');
        logoImg.src = 'logo.svg';
        logoImg.alt = 'Agentic Guide to AI Agents';
        
        logoDiv.appendChild(logoImg);
        
        // Insert at the beginning of sidebar
        sidebar.insertBefore(logoDiv, sidebar.firstChild);
    });
})();
