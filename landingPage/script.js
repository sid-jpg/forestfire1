document.addEventListener('DOMContentLoaded', function() {
    // Get the predict button
    const predictBtn = document.querySelector('.predict-btn');
    
    // Add click event listener
    predictBtn.addEventListener('click', function() {
        // Redirect to the main application page
        window.location.href = '/';  // Updated to use the root route
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
});