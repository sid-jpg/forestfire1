// Landing page interactions
document.addEventListener('DOMContentLoaded', function() {
    // Mode selection functionality
    window.showModeSelection = function() {
        const modal = document.getElementById('modeSelection');
        modal.style.display = 'flex';
        setTimeout(() => {
            modal.classList.add('active');
        }, 10);
    };

    // Let's Save button click handler
    document.querySelector('.predict-btn').addEventListener('click', function() {
        window.showModeSelection();
    });

    // Add hover animation to leaves
    const leaves = document.querySelectorAll('.fixed-leaf');
    leaves.forEach(leaf => {
        leaf.addEventListener('mouseover', () => {
            leaf.style.transform = 'rotate(10deg) scale(1.1)';
        });
        leaf.addEventListener('mouseout', () => {
            leaf.style.transform = 'rotate(0deg) scale(1)';
        });
    });

    // Navbar background change on scroll
    window.addEventListener('scroll', function() {
        const navbar = document.querySelector('.navbar');
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    });

    // Close modal when clicking outside
    document.addEventListener('click', function(event) {
        const modal = document.getElementById('modeSelection');
        const modalContent = document.querySelector('.mode-content');
        if (event.target === modal) {
            modal.classList.remove('active');
            setTimeout(() => {
                modal.style.display = 'none';
            }, 300);
        }
    });

    // Add hover effects to mode options
    const modeOptions = document.querySelectorAll('.mode-option');
    modeOptions.forEach(option => {
        option.addEventListener('mouseenter', function() {
            this.classList.add('hover');
        });
        option.addEventListener('mouseleave', function() {
            this.classList.remove('hover');
        });
    });
});
