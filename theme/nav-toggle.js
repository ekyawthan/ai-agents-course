// Enhanced navigation with double-tap expand/collapse
(function() {
    document.addEventListener('DOMContentLoaded', function() {
        // Wait for sidebar to be ready
        setTimeout(initNavToggle, 100);
    });
    
    function initNavToggle() {
        const sidebar = document.querySelector('.sidebar-scrollbox');
        if (!sidebar) return;
        
        const chapterItems = sidebar.querySelectorAll('li.chapter-item');
        
        chapterItems.forEach(item => {
            const link = item.querySelector('a');
            if (!link) return;
            
            // Check if item has sub-chapters (ol element)
            const subList = item.querySelector('ol');
            if (!subList) return;
            
            let lastTap = 0;
            
            link.addEventListener('click', function(e) {
                const now = Date.now();
                const timeSinceLastTap = now - lastTap;
                
                if (timeSinceLastTap < 300 && timeSinceLastTap > 0) {
                    // Double tap detected
                    e.preventDefault();
                    e.stopPropagation();
                    
                    // Toggle visibility
                    if (subList.style.display === 'none') {
                        subList.style.display = 'block';
                        item.classList.add('expanded');
                    } else {
                        subList.style.display = 'none';
                        item.classList.remove('expanded');
                    }
                    
                    lastTap = 0; // Reset
                } else {
                    // First tap
                    lastTap = now;
                }
            });
        });
    }
})();
