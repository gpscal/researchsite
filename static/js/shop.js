// Shop page JavaScript functionality

document.addEventListener('DOMContentLoaded', function() {
    // Handle navigation arrows in "What's New" section
    const prevButton = document.querySelector('.nav-arrow.prev');
    const nextButton = document.querySelector('.nav-arrow.next');
    const viewAllButton = document.querySelector('.view-all-btn');

    // Navigation arrow functionality
    if (prevButton) {
        prevButton.addEventListener('click', function() {
            console.log('Previous button clicked');
            // Add your previous slide logic here
        });
    }

    if (nextButton) {
        nextButton.addEventListener('click', function() {
            console.log('Next button clicked');
            // Add your next slide logic here
        });
    }

    if (viewAllButton) {
        viewAllButton.addEventListener('click', function() {
            console.log('View all button clicked');
            // Add your view all logic here
        });
    }

    // Shop Now button functionality
    const shopNowButton = document.querySelector('.shop-now-btn');
    if (shopNowButton) {
        shopNowButton.addEventListener('click', function() {
            // Scroll to product grid or navigate to shop section
            const productGrid = document.querySelector('.product-grid');
            if (productGrid) {
                productGrid.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    }

    // Add hover effects for product cards
    const productCards = document.querySelectorAll('.product-card');
    productCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-8px)';
        });

        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });

    // Add loading animation for images
    const images = document.querySelectorAll('img');
    images.forEach(img => {
        img.addEventListener('load', function() {
            this.style.opacity = '1';
        });

        // Set initial opacity for fade-in effect
        img.style.opacity = '0';
        img.style.transition = 'opacity 0.3s ease';
    });

    // Add parallax effect to hero section
    let ticking = false;

    function updateParallax() {
        const scrolled = window.pageYOffset;
        const parallax = document.querySelector('.hero-image img');
        if (parallax) {
            const speed = scrolled * 0.5;
            parallax.style.transform = `translateY(${speed}px)`;
        }
        ticking = false;
    }

    function requestTick() {
        if (!ticking) {
            requestAnimationFrame(updateParallax);
            ticking = true;
        }
    }

    window.addEventListener('scroll', requestTick);
});
