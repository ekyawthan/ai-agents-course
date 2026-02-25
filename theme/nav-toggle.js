// Enhanced navigation with double-tap expand/collapse
(function() {
    document.addEventListener('DOMContentLoaded', function() {
        const chapterItems = document.querySelectorAll('.chapter-item');
        
        chapterItems.forEach(item => {
            const link = item.querySelector('a');
            if (!link) return;
            
            // Check if this item has children
            const hasChildren = item.classList.contains('expanded') || 
                               item.querySelector('ol') !== null;
            
            if (hasChildren) {
                let tapCount = 0;
                let tapTimer = null;
                
                link.addEventListener('click', function(e) {
                    tapCount++;
                    
                    if (tapCount === 1) {
                        // First tap - wait for potential second tap
                        tapTimer = setTimeout(() => {
                            tapCount = 0;
                            // Single tap - default navigation
                        }, 300);
                    } else if (tapCount === 2) {
                        // Double tap - toggle expand/collapse
                        clearTimeout(tapTimer);
                        tapCount = 0;
                        
                        e.preventDefault();
                        e.stopPropagation();
                        
                        // Toggle expanded class
                        if (item.classList.contains('expanded')) {
                            item.classList.remove('expanded');
                        } else {
                            item.classList.add('expanded');
                        }
                    }
                });
            }
        });
    });
})();
